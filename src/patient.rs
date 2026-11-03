use anyhow::{anyhow, bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen3::{Config, ModelForCausalLM};
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{HashMap, HashSet},
    fs,
    io::Read,
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, Ordering},
    time::{Instant, SystemTime, UNIX_EPOCH},
};
use tokenizers::Tokenizer;

const MODEL: &str = "Qwen/Qwen3-0.6B";
const REVISION: &str = "c1899de289a04d12100db370d81485cdf75e47ca";
const MAX_EDIT: usize = 262_144;

pub struct Patient {
    data_dir: PathBuf,
    device: Device,
    device_name: String,
    config: Config,
    tokenizer: Tokenizer,
    baseline: HashMap<String, Tensor>,
    working: HashMap<String, Tensor>,
    changed: HashMap<String, usize>,
    model: ModelForCausalLM,
    parameters: usize,
    base_hash: String,
    baseline_path: PathBuf,
}

fn model_from(
    weights: &HashMap<String, Tensor>,
    cfg: &Config,
    device: &Device,
) -> Result<ModelForCausalLM> {
    let dtype = if device.is_cpu() {
        DType::F32
    } else {
        DType::BF16
    };
    let vb = VarBuilder::from_tensors(weights.clone(), dtype, device);
    Ok(ModelForCausalLM::new(cfg, vb)?)
}
fn file_hash(path: &Path) -> Result<String> {
    let mut f = fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = [0u8; 1024 * 1024];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(format!("{:x}", h.finalize()))
}
fn usize_arg(v: &Value, key: &str, default: usize) -> Result<usize> {
    match v.get(key) {
        None => Ok(default),
        Some(x) => x
            .as_u64()
            .and_then(|n| usize::try_from(n).ok())
            .ok_or_else(|| anyhow!("{key} must be a non-negative integer")),
    }
}
fn selection_indices(shape: &[usize], sel: &Value) -> Result<Vec<usize>> {
    let kind = sel
        .get("kind")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("selection.kind is required"))?;
    let start = usize_arg(sel, "start", 0)?;
    let count = usize_arg(sel, "count", 0)?;
    if count == 0 {
        bail!("selection.count must be positive")
    }
    let total = shape
        .iter()
        .try_fold(1usize, |a, &b| a.checked_mul(b))
        .ok_or_else(|| anyhow!("shape overflow"))?;
    let end = start
        .checked_add(count)
        .ok_or_else(|| anyhow!("selection overflow"))?;
    match kind {
        "range" => {
            if end > total || count > MAX_EDIT {
                bail!("range exceeds tensor bounds or edit limit ({MAX_EDIT})")
            }
            Ok((start..end).collect())
        }
        "rows" | "columns" => {
            if shape.len() != 2 {
                bail!("rows and columns require a 2D tensor")
            }
            let (rows, cols) = (shape[0], shape[1]);
            let (bound, width) = if kind == "rows" {
                (rows, cols)
            } else {
                (cols, rows)
            };
            if end > bound
                || count
                    .checked_mul(width)
                    .filter(|&x| x <= MAX_EDIT)
                    .is_none()
            {
                bail!("selection exceeds tensor bounds or edit limit ({MAX_EDIT})")
            }
            if kind == "rows" {
                Ok((start * cols..end * cols).collect())
            } else {
                Ok((0..rows)
                    .flat_map(|r| (start..end).map(move |c| r * cols + c))
                    .collect())
            }
        }
        _ => bail!("selection.kind must be range, rows, or columns"),
    }
}
fn flat_values(t: &Tensor) -> Result<Vec<f32>> {
    Ok(t.to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?)
}
fn count_diff(a: &Tensor, b: &Tensor) -> Result<usize> {
    let a = flat_values(a)?;
    let b = flat_values(b)?;
    Ok(a.iter()
        .zip(&b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count())
}
impl Patient {
    pub fn load(data_dir: &Path, requested_device: &str) -> Result<Self> {
        fs::create_dir_all(data_dir)?;
        let (device, device_name) = match requested_device {
            "auto" => {
                #[cfg(feature = "metal")]
                {
                    match Device::new_metal(0) {
                        Ok(device) => (device, "metal".to_owned()),
                        Err(_) => (Device::Cpu, "cpu".to_owned()),
                    }
                }
                #[cfg(not(feature = "metal"))]
                {
                    (Device::Cpu, "cpu".to_owned())
                }
            }
            "cpu" => (Device::Cpu, "cpu".to_owned()),
            "metal" => {
                #[cfg(feature = "metal")]
                {
                    (Device::new_metal(0)?, "metal".to_owned())
                }
                #[cfg(not(feature = "metal"))]
                {
                    bail!("this build does not support Metal")
                }
            }
            _ => bail!("device must be auto, cpu, or metal"),
        };
        let api = ApiBuilder::new()
            .with_cache_dir(data_dir.join("huggingface"))
            .build()?;
        let repo = api.repo(Repo::with_revision(
            MODEL.to_owned(),
            RepoType::Model,
            REVISION.to_owned(),
        ));
        let config_path = repo.get("config.json").context("fetching model config")?;
        let tokenizer_path = repo.get("tokenizer.json").context("fetching tokenizer")?;
        let weights_path = repo
            .get("model.safetensors")
            .context("fetching model weights")?;
        let config: Config = serde_json::from_slice(&fs::read(config_path)?)?;
        let tokenizer =
            Tokenizer::from_file(tokenizer_path).map_err(|e| anyhow!("tokenizer: {e}"))?;
        let base_hash = file_hash(&weights_path)?;
        let mut baseline = candle_core::safetensors::load(&weights_path, &device)
            .context("loading safetensors")?;
        // Candle ties the output projection to the input embedding. The upstream
        // checkpoint also stores a duplicate lm_head that Candle never reads.
        if config.tie_word_embeddings {
            baseline.remove("lm_head.weight");
        }
        let parameters = baseline.values().map(Tensor::elem_count).sum();
        let working = baseline.clone();
        let model = model_from(&working, &config, &device)?;
        Ok(Self {
            data_dir: data_dir.to_owned(),
            device,
            device_name,
            config,
            tokenizer,
            baseline,
            working,
            changed: HashMap::new(),
            model,
            parameters,
            base_hash,
            baseline_path: weights_path,
        })
    }
    pub fn status(&self) -> Value {
        let changed_parameters: usize = self.changed.values().sum();
        let mut modified_tensors: Vec<_> = self.changed.keys().cloned().collect();
        modified_tensors.sort();
        json!({"model":MODEL,"revision":REVISION,"loaded":true,"parameters":self.parameters,"changed_parameters":changed_parameters,"changed_percent":100.0 * changed_parameters as f64 / self.parameters as f64,"layers":self.config.num_hidden_layers,"modified_tensors":modified_tensors,"device":self.device_name,"base_sha256":self.base_hash,"source_file":self.baseline_path,"source_dtype":"BF16","inference_dtype":if self.device.is_cpu(){"F32"}else{"BF16"}})
    }
    pub fn inspect_model(&self, args: Value) -> Result<Value> {
        let offset = usize_arg(&args, "offset", 0)?;
        let limit = usize_arg(&args, "limit", 40)?.min(200);
        let layer = args.get("layer").and_then(Value::as_u64);
        let mut names: Vec<_> = self
            .working
            .keys()
            .filter(|n| layer.map_or(true, |l| n.starts_with(&format!("model.layers.{l}."))))
            .collect();
        names.sort();
        let total = names.len();
        let tensors: Vec<Value> = names.into_iter().skip(offset).take(limit).map(|n| -> Result<Value> {
            let t=&self.working[n]; let count=t.elem_count().min(64);
            let sample=t.flatten_all()?.narrow(0,0,count)?.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
            let stats=if sample.is_empty() {json!({"scope":"first values","count":0})} else {json!({"scope":"first values","count":count,"min":sample.iter().copied().fold(f32::INFINITY,f32::min),"max":sample.iter().copied().fold(f32::NEG_INFINITY,f32::max),"mean":sample.iter().map(|&x|x as f64).sum::<f64>()/count as f64})};
            Ok(json!({"name":n,"shape":t.dims(),"dtype":format!("{:?}",t.dtype()),"parameters":t.elem_count(),"changed_parameters":self.changed.get(n).copied().unwrap_or(0),"sample_stats":stats}))
        }).collect::<Result<_>>()?;
        Ok(
            json!({"status":self.status(),"architecture":{"hidden_size":self.config.hidden_size,"intermediate_size":self.config.intermediate_size,"heads":self.config.num_attention_heads,"kv_heads":self.config.num_key_value_heads,"head_dim":self.config.head_dim,"vocab_size":self.config.vocab_size,"max_position_embeddings":self.config.max_position_embeddings,"tie_word_embeddings":self.config.tie_word_embeddings,"tied_output_head":"model.embed_tokens.weight"},"tensors":tensors,"total":total,"offset":offset,"limit":limit}),
        )
    }
    pub fn inspect_weights(&self, args: Value) -> Result<Value> {
        let name = args
            .get("tensor")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("tensor is required"))?;
        let t = self
            .working
            .get(name)
            .ok_or_else(|| anyhow!("unknown tensor"))?;
        let offset = usize_arg(&args, "offset", 0)?;
        let count = usize_arg(&args, "count", 64)?.min(1024);
        if offset > t.elem_count() {
            bail!("offset exceeds tensor bounds")
        }
        let count = count.min(t.elem_count() - offset);
        let (slice, selected_count) = if let Some(selection) = args.get("selection") {
            let indices = selection_indices(t.dims(), selection)?;
            let selected_count = indices.len();
            let sample: Vec<u32> = indices
                .iter()
                .take(1024)
                .map(|&i| u32::try_from(i))
                .collect::<std::result::Result<_, _>>()?;
            let ids = Tensor::new(sample.as_slice(), &self.device)?;
            (
                t.flatten_all()?
                    .index_select(&ids, 0)?
                    .to_device(&Device::Cpu)?
                    .to_dtype(DType::F32)?
                    .to_vec1::<f32>()?,
                selected_count,
            )
        } else {
            (
                t.flatten_all()?
                    .narrow(0, offset, count)?
                    .to_device(&Device::Cpu)?
                    .to_dtype(DType::F32)?
                    .to_vec1::<f32>()?,
                count,
            )
        };
        let (min, max, mean) = if slice.is_empty() {
            (None, None, None)
        } else {
            (
                Some(slice.iter().copied().fold(f32::INFINITY, f32::min)),
                Some(slice.iter().copied().fold(f32::NEG_INFINITY, f32::max)),
                Some(slice.iter().map(|&x| x as f64).sum::<f64>() / slice.len() as f64),
            )
        };
        Ok(
            json!({"tensor":name,"shape":t.dims(),"dtype":format!("{:?}",t.dtype()),"offset":offset,"selected_count":selected_count,"values":slice,"sample_stats":{"min":min,"max":max,"mean":mean}}),
        )
    }
    pub fn edit_weights(&mut self, args: Value) -> Result<Value> {
        let name = args
            .get("tensor")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("tensor is required"))?;
        let tensor = self
            .working
            .get(name)
            .ok_or_else(|| anyhow!("unknown tensor"))?;
        let indices = selection_indices(
            tensor.dims(),
            args.get("selection")
                .ok_or_else(|| anyhow!("selection is required"))?,
        )?;
        let op = args
            .get("operation")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("operation is required"))?;
        if !["scale", "zero", "replace"].contains(&op) {
            bail!("operation must be scale, zero, or replace")
        }
        let value = if op == "zero" {
            0.0
        } else {
            args.get("value")
                .and_then(Value::as_f64)
                .ok_or_else(|| anyhow!("finite value is required"))?
        };
        if !value.is_finite() || !(value as f32).is_finite() {
            bail!("value must be finite and representable")
        }
        let mut data = flat_values(tensor)?;
        for &i in &indices {
            let x = match op {
                "scale" => (data[i] as f64) * value,
                "zero" => 0.0,
                _ => value,
            };
            if !x.is_finite() || !(x as f32).is_finite() {
                bail!("edit would produce non-finite values")
            }
            data[i] = x as f32;
        }
        let edited = Tensor::from_vec(data, tensor.dims(), &Device::Cpu)?
            .to_dtype(tensor.dtype())?
            .to_device(&self.device)?;
        if flat_values(&edited)?.iter().any(|v| !v.is_finite()) {
            bail!("edit is not representable in tensor dtype")
        }
        let changed = count_diff(&edited, &self.baseline[name])?;
        let mut candidate = self.working.clone();
        if changed == 0 {
            candidate.insert(name.to_owned(), self.baseline[name].clone());
        } else {
            candidate.insert(name.to_owned(), edited);
        }
        let model = model_from(&candidate, &self.config, &self.device)?;
        self.working = candidate;
        self.model = model;
        if changed == 0 {
            self.changed.remove(name);
        } else {
            self.changed.insert(name.to_owned(), changed);
        }
        Ok(
            json!({"tensor":name,"selected_parameters":indices.len(),"changed_parameters":changed,"status":self.status()}),
        )
    }
    pub fn revert_changes(&mut self) -> Result<Value> {
        let model = model_from(&self.baseline, &self.config, &self.device)?;
        self.working = self.baseline.clone();
        self.changed.clear();
        self.model = model;
        Ok(self.status())
    }
    pub fn save_patch(&self) -> Result<Value> {
        if self.changed.is_empty() {
            bail!("there are no changes to save")
        }
        let root = self.data_dir.join("patches");
        fs::create_dir_all(&root)?;
        let id = SystemTime::now()
            .duration_since(UNIX_EPOCH)?
            .as_nanos()
            .to_string();
        let dir = root.join(&id);
        fs::create_dir(&dir)?;
        let weights: HashMap<String, Tensor> = self
            .changed
            .keys()
            .map(|n| (n.clone(), self.working[n].clone()))
            .collect();
        let path = dir.join("weights.safetensors");
        candle_core::safetensors::save(&weights, &path)?;
        let mut names: Vec<_> = self.changed.keys().cloned().collect();
        names.sort();
        let manifest = json!({"version":1,"model":MODEL,"revision":REVISION,"base_sha256":self.base_hash,"weights_sha256":file_hash(&path)?,"tensors":names.iter().map(|n|json!({"name":n,"shape":self.working[n].dims(),"dtype":format!("{:?}",self.working[n].dtype()),"changed_parameters":self.changed[n]})).collect::<Vec<_>>()});
        fs::write(
            dir.join("manifest.json"),
            serde_json::to_vec_pretty(&manifest)?,
        )?;
        Ok(json!({"id":id,"path":dir,"manifest":manifest}))
    }
    pub fn apply_patch(&mut self, args: Value) -> Result<Value> {
        let id = args
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("patch id required"))?;
        if id.is_empty() || !id.bytes().all(|c| c.is_ascii_digit()) {
            bail!("invalid patch id")
        }
        let dir = self.data_dir.join("patches").join(id);
        let manifest: Value = serde_json::from_slice(&fs::read(dir.join("manifest.json"))?)?;
        if manifest["version"] != 1
            || manifest["model"] != MODEL
            || manifest["revision"] != REVISION
            || manifest["base_sha256"] != self.base_hash
        {
            bail!("patch does not match this baseline")
        }
        let path = dir.join("weights.safetensors");
        if manifest["weights_sha256"] != file_hash(&path)? {
            bail!("patch checksum mismatch")
        }
        let tensors = candle_core::safetensors::load(&path, &self.device)?;
        let listed = manifest["tensors"]
            .as_array()
            .ok_or_else(|| anyhow!("invalid manifest"))?;
        if listed.len() != tensors.len() {
            bail!("patch tensor list mismatch")
        }
        let mut candidate = self.baseline.clone();
        let mut changed = HashMap::new();
        let mut seen = HashSet::new();
        for entry in listed {
            let n = entry["name"]
                .as_str()
                .ok_or_else(|| anyhow!("invalid tensor name"))?;
            if !seen.insert(n) {
                bail!("duplicate tensor")
            };
            let t = tensors
                .get(n)
                .ok_or_else(|| anyhow!("missing patch tensor"))?;
            let b = self
                .baseline
                .get(n)
                .ok_or_else(|| anyhow!("unknown patch tensor"))?;
            if t.dims() != b.dims()
                || t.dtype() != b.dtype()
                || flat_values(t)?.iter().any(|v| !v.is_finite())
            {
                bail!("invalid patch tensor")
            };
            let count = count_diff(t, b)?;
            if entry["changed_parameters"].as_u64() != Some(count as u64) {
                bail!("patch changed count mismatch")
            };
            if count > 0 {
                candidate.insert(n.to_owned(), t.clone());
                changed.insert(n.to_owned(), count);
            }
        }
        let model = model_from(&candidate, &self.config, &self.device)?;
        self.working = candidate;
        self.changed = changed;
        self.model = model;
        Ok(self.status())
    }
    fn generate(
        &mut self,
        args: &Value,
        baseline: bool,
        cancel: &AtomicBool,
    ) -> Result<Vec<Value>> {
        let prompts = args
            .get("prompts")
            .and_then(Value::as_array)
            .ok_or_else(|| anyhow!("prompts must be an array"))?;
        if prompts.is_empty() || prompts.len() > 8 {
            bail!("provide between 1 and 8 prompts")
        }
        let max_tokens = usize_arg(args, "max_tokens", 64)?;
        if !(1..=512).contains(&max_tokens) {
            bail!("max_tokens must be between 1 and 512")
        }
        let thinking = args
            .get("thinking")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let mut base_model = if baseline {
            Some(model_from(&self.baseline, &self.config, &self.device)?)
        } else {
            None
        };
        let model = if let Some(m) = base_model.as_mut() {
            m
        } else {
            &mut self.model
        };
        let mut rows = Vec::new();
        for prompt in prompts {
            if cancel.load(Ordering::Relaxed) {
                bail!("generation cancelled")
            }
            let prompt = prompt
                .as_str()
                .ok_or_else(|| anyhow!("each prompt must be a string"))?;
            if prompt.len() > 16_384 {
                bail!("prompt exceeds 16384 bytes")
            }
            let formatted = format!(
                "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n{}",
                prompt.trim(),
                if thinking {
                    ""
                } else {
                    "<think>\n\n</think>\n\n"
                }
            );
            let input = self
                .tokenizer
                .encode(formatted.as_str(), false)
                .map_err(|e| anyhow!("tokenization: {e}"))?;
            let tokens = input.get_ids();
            if tokens.is_empty()
                || tokens.len() + max_tokens > self.config.max_position_embeddings
                || tokens.len() > 2048
            {
                bail!("prompt exceeds token limit")
            }
            model.clear_kv_cache();
            let start = Instant::now();
            let mut generated = Vec::new();
            let mut next_input = tokens.to_vec();
            let mut offset = 0;
            let mut hit = true;
            for _ in 0..max_tokens {
                if cancel.load(Ordering::Relaxed) {
                    bail!("generation cancelled")
                }
                let input = Tensor::new(next_input.as_slice(), &self.device)?.unsqueeze(0)?;
                let logits = model
                    .forward(&input, offset)?
                    .squeeze(0)?
                    .squeeze(0)?
                    .to_dtype(DType::F32)?;
                offset += next_input.len();
                let next = logits.argmax(0)?.to_scalar::<u32>()?;
                if next == 151645 || next == 151643 {
                    hit = false;
                    break;
                }
                generated.push(next);
                next_input = vec![next];
            }
            let output = self
                .tokenizer
                .decode(&generated, true)
                .map_err(|e| anyhow!("decoding: {e}"))?;
            let mut seen = HashSet::new();
            let repeated = generated.iter().filter(|n| !seen.insert(**n)).count();
            rows.push(json!({"prompt":prompt,"output":output,"tokens":generated.len(),"elapsed_ms":start.elapsed().as_millis(),"hit_token_limit":hit,"repetition_rate":if generated.is_empty(){0.0}else{repeated as f64/generated.len() as f64}}));
        }
        Ok(rows)
    }
    pub fn test_patient(&mut self, args: Value, cancel: &AtomicBool) -> Result<Value> {
        let rows = self.generate(&args, false, cancel)?;
        Ok(
            json!({"rows":rows,"settings":{"decoding":"greedy","thinking":args.get("thinking").and_then(Value::as_bool).unwrap_or(false),"max_tokens":usize_arg(&args,"max_tokens",64)?}}),
        )
    }
    pub fn compare_patient(&mut self, args: Value, cancel: &AtomicBool) -> Result<Value> {
        let baseline = self.generate(&args, true, cancel)?;
        let working = self.generate(&args, false, cancel)?;
        Ok(
            json!({"baseline":baseline,"working":working,"settings":{"decoding":"greedy","thinking":args.get("thinking").and_then(Value::as_bool).unwrap_or(false),"max_tokens":usize_arg(&args,"max_tokens",64)?}}),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn selections_are_bounded_and_row_major() {
        assert_eq!(
            selection_indices(&[2, 4], &json!({"kind":"columns","start":1,"count":2})).unwrap(),
            vec![1, 2, 5, 6]
        );
        assert_eq!(
            selection_indices(&[2, 4], &json!({"kind":"rows","start":1,"count":1})).unwrap(),
            vec![4, 5, 6, 7]
        );
        assert!(selection_indices(&[2, 4], &json!({"kind":"range","start":7,"count":2})).is_err());
        assert!(
            selection_indices(&[1024, 1024], &json!({"kind":"rows","start":0,"count":257}))
                .is_err()
        );
        assert!(selection_indices(
            &[2, 4],
            &json!({"kind":"columns","start":usize::MAX,"count":2})
        )
        .is_err());
    }
    #[test]
    #[ignore = "downloads the real 1.5 GB Qwen3 model and runs CPU inference"]
    fn real_model_edit_revert() -> Result<()> {
        let dir = std::env::var_os("OPERATING_TABLE_TEST_DATA")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("data"));
        let mut p = Patient::load(&dir, "cpu")?;
        assert_eq!(p.parameters, 596_049_920);
        assert!(!p.working.contains_key("lm_head.weight"));
        let original_hash = p.base_hash.clone();
        let name = "model.layers.0.mlp.down_proj.weight";
        let before = flat_values(&p.working[name])?;
        p.edit_weights(json!({"tensor":name,"selection":{"kind":"range","start":0,"count":8},"operation":"zero"}))?;
        let edited = flat_values(&p.working[name])?;
        assert_ne!(before, edited);
        let changed = p.changed[name];
        assert!(changed > 0);
        assert!(p.edit_weights(json!({"tensor":name,"selection":{"kind":"range","start":before.len()-1,"count":2},"operation":"replace","value":1})).is_err());
        assert_eq!(edited, flat_values(&p.working[name])?);
        assert_eq!(changed, p.changed[name]);
        let cancel = AtomicBool::new(false);
        let args = json!({"prompts":["What is 2 + 2?"],"max_tokens":4});
        let compared = p.compare_patient(args.clone(), &cancel)?;
        let patch = p.save_patch()?;
        p.revert_changes()?;
        assert_eq!(before, flat_values(&p.working[name])?);
        let restored = p.test_patient(args.clone(), &cancel)?;
        assert_eq!(
            compared["baseline"][0]["output"],
            restored["rows"][0]["output"]
        );
        p.apply_patch(json!({"id":patch["id"]}))?;
        assert_eq!(edited, flat_values(&p.working[name])?);
        let replay = p.test_patient(args, &cancel)?;
        assert_eq!(
            compared["working"][0]["output"],
            replay["rows"][0]["output"]
        );
        p.revert_changes()?;
        assert_eq!(original_hash, file_hash(&p.baseline_path)?);
        Ok(())
    }
}
