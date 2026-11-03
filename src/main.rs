mod patient;
mod surgeon;

use anyhow::{anyhow, Context, Result};
use axum::{
    extract::{DefaultBodyLimit, Request, State},
    http::{header, HeaderMap, StatusCode},
    middleware::{self, Next},
    response::{Html, IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::Serialize;
use serde_json::{json, Value};
use std::{
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    time::{SystemTime, UNIX_EPOCH},
};

const EVALUATION: [&str; 3] = [
    "What is 17 + 26? Give just the answer.",
    "A train travels 60 km in 45 minutes. What is its average speed in km/h? Explain briefly.",
    "Write exactly three different words that name colors, separated by commas.",
];

#[derive(Serialize)]
struct Room {
    phase: String,
    busy: bool,
    patient: Value,
    activity: Vec<Value>,
    comparison: Value,
    error: Option<String>,
    active_tensor: Option<String>,
    tool_calls: usize,
    experiments: usize,
    limits: Value,
    surgeon: Value,
}
#[derive(Clone)]
struct App {
    patient: Arc<Mutex<Option<patient::Patient>>>,
    room: Arc<Mutex<Room>>,
    busy: Arc<AtomicBool>,
    cancel: Arc<AtomicBool>,
    data: Arc<PathBuf>,
    device: Arc<String>,
    port: u16,
}
impl App {
    fn event(&self, kind: &str, message: impl Into<String>, data: Value) {
        let mut room = self.room.lock().unwrap();
        let id = room
            .activity
            .last()
            .and_then(|v| v["id"].as_u64())
            .unwrap_or(0)
            + 1;
        room.activity
            .push(json!({"id":id,"kind":kind,"message":message.into(),"data":data}));
        if room.activity.len() > 250 {
            room.activity.remove(0);
        }
    }
    fn check_cancel(&self) -> Result<()> {
        if self.cancel.load(Ordering::SeqCst) {
            anyhow::bail!("Procedure cancelled")
        }
        Ok(())
    }
    fn claim(&self, phase: &str) -> Result<()> {
        let mut room = self.room.lock().unwrap();
        self.busy
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .map_err(|_| anyhow!("An operation is already running"))?;
        self.cancel.store(false, Ordering::SeqCst);
        room.busy = true;
        room.phase = phase.into();
        room.error = None;
        Ok(())
    }
    fn settle(&self, room: &mut Room, outcome: Result<()>) {
        match outcome {
            Ok(()) => {
                room.phase = if room.patient.is_null() {
                    "idle"
                } else {
                    "ready"
                }
                .into()
            }
            Err(e) => {
                room.phase = "error".into();
                room.error = Some(format!("{e:#}"));
            }
        }
        room.busy = false;
        room.active_tensor = None;
        self.busy.store(false, Ordering::SeqCst);
    }
    async fn finish(&self, outcome: Result<()>) {
        // claim, cancellation and release share this lock; a late revert cannot be lost.
        let loaded = {
            let mut room = self.room.lock().unwrap();
            if !self.cancel.load(Ordering::SeqCst) {
                self.settle(&mut room, outcome);
                return;
            }
            room.phase = "reverting".into();
            !room.patient.is_null()
        };
        let result = if loaded {
            let result = self.call("revert_changes", json!({})).await.map(|_| ());
            if result.is_ok() {
                self.event(
                    "system",
                    "Procedure stopped. Original weights restored.",
                    json!({}),
                );
            }
            result
        } else {
            self.event(
                "system",
                "Procedure stopped; no patient was loaded.",
                json!({}),
            );
            outcome
        };
        let mut room = self.room.lock().unwrap();
        self.settle(&mut room, result);
    }
    async fn call(&self, tool: &str, arguments: Value) -> Result<Value> {
        if tool != "revert_changes" {
            self.check_cancel()?;
        }
        if let Some(tensor) = arguments["tensor"].as_str() {
            self.room.lock().unwrap().active_tensor = Some(tensor.into());
        }
        self.event("tool", format!("Using {tool}"), arguments.clone());
        let app = self.clone();
        let name = tool.to_string();
        let result = tokio::task::spawn_blocking(move || -> Result<Value> {
            let mut guard = app.patient.lock().unwrap();
            let p = guard.as_mut().context("Load the patient first")?;
            let result = match name.as_str() {
                "inspect_model" => p.inspect_model(arguments),
                "inspect_weights" => p.inspect_weights(arguments),
                "test_patient" => p.test_patient(arguments, &app.cancel),
                "edit_weights" => p.edit_weights(arguments),
                "compare_patient" => p.compare_patient(arguments, &app.cancel),
                "revert_changes" => p.revert_changes(),
                "save_patch" => p.save_patch(),
                "apply_patch" => p.apply_patch(arguments),
                _ => Err(anyhow!("Unknown surgical tool: {name}")),
            };
            app.room.lock().unwrap().patient = p.status();
            result
        })
        .await
        .context("Patient worker failed")?;
        match &result {
            Ok(value) => {
                if tool == "compare_patient" {
                    self.room.lock().unwrap().comparison = value.clone();
                }
                if tool == "revert_changes" {
                    self.room.lock().unwrap().comparison = Value::Null;
                }
                self.event(
                    if tool == "edit_weights" {
                        "edit"
                    } else if tool == "compare_patient" {
                        "comparison"
                    } else {
                        "result"
                    },
                    format!("{tool} completed"),
                    value.clone(),
                );
            }
            Err(error) => self.event("error", format!("{tool}: {error:#}"), json!({})),
        }
        result
    }
    fn eval_args(&self) -> Value {
        json!({"prompts":EVALUATION,"max_tokens":64,"thinking":false})
    }
    async fn operate(&self, objective: String, calls: usize, edits: usize) -> Result<()> {
        self.event(
            "system",
            "Recording the immutable baseline and current working model on three fixed prompts.",
            self.eval_args(),
        );
        let initial = self.call("compare_patient", self.eval_args()).await?;
        let mut history =
            vec![json!({"tool":"compare_patient","arguments":self.eval_args(),"result":initial})];
        let mut used_edits = 0;
        let investigation: Result<()> = async {
        for index in 0..calls {
            self.check_cancel()?;
            self.event("system", "Astra is examining the evidence through Codex.", json!({"turn":index+1}));
            let action = surgeon::next_action(&objective, &history, calls-index, edits-used_edits, self.cancel.clone(), &self.data).await?;
            self.check_cancel()?;
            self.event("observation", &action.summary, json!({"tool":action.tool,"arguments":action.arguments}));
            if action.tool == "finish" { break; }
            self.room.lock().unwrap().tool_calls = index+1;
            let result = if action.tool == "edit_weights" {
                if used_edits >= edits { Err(anyhow!("Experiment budget exhausted; compare or finish")) }
                else {
                    used_edits += 1;
                    self.room.lock().unwrap().experiments = used_edits;
                    self.call(&action.tool, action.arguments.clone()).await
                }
            } else { self.call(&action.tool, action.arguments.clone()).await };
            history.push(json!({"summary":action.summary,"tool":action.tool,"arguments":action.arguments,"result":match result {Ok(v)=>v,Err(e)=>json!({"error":format!("{e:#}")})}}));
        }
        Ok(())
        }.await;
        self.check_cancel()?;
        if let Err(error) = &investigation {
            self.event("error", format!("Investigation stopped: {error:#}. Recording the current working copy for review."), json!({}));
        }
        self.event("system", "Running the final matched comparison. A successful edit does not establish an improvement.", json!({}));
        self.call("compare_patient", self.eval_args()).await?;
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis();
        let path = self.data.join("operations");
        std::fs::create_dir_all(&path)?;
        let room = self.room.lock().unwrap();
        std::fs::write(
            path.join(format!("{timestamp}.json")),
            serde_json::to_vec_pretty(
                &json!({"objective":objective,"history":history,"comparison":room.comparison,"limits":room.limits,"error":investigation.as_ref().err().map(|e|format!("{e:#}"))}),
            )?,
        )?;
        drop(room);
        self.event(
            "system",
            "Procedure finished. Review the comparison, then save the patch or revert.",
            json!({}),
        );
        investigation
    }
}

fn api_error(status: StatusCode, error: impl ToString) -> Response {
    (status, Json(json!({"error":error.to_string()}))).into_response()
}
async fn state(State(app): State<App>) -> Json<Value> {
    Json(serde_json::to_value(&*app.room.lock().unwrap()).unwrap())
}
async fn load(State(app): State<App>, Json(_): Json<Value>) -> Response {
    if let Err(e) = app.claim("loading") {
        return api_error(StatusCode::CONFLICT, e);
    }
    tokio::spawn(async move {
        app.event("system", "Loading original Qwen3-0.6B weights. The first load downloads about 1.2 GB from Hugging Face.", json!({}));
        let worker = app.clone();
        let result = tokio::task::spawn_blocking(move || -> Result<()> {
            let patient = patient::Patient::load(&worker.data, &worker.device)?;
            worker.room.lock().unwrap().patient = patient.status();
            *worker.patient.lock().unwrap() = Some(patient);
            Ok(())
        })
        .await
        .map_err(anyhow::Error::from)
        .and_then(|v| v);
        if result.is_ok() {
            app.event(
                "system",
                "Patient loaded. Original weights preserved; working copy ready.",
                json!({}),
            );
        }
        app.finish(result).await;
    });
    Json(json!({"ok":true})).into_response()
}
async fn surgery(State(app): State<App>, Json(args): Json<Value>) -> Response {
    let objective = args["objective"].as_str().unwrap_or("").trim().to_owned();
    if objective.is_empty() || objective.len() > 4000 {
        return api_error(
            StatusCode::BAD_REQUEST,
            "Enter an objective of 1–4000 bytes",
        );
    }
    let calls = args["max_tool_calls"].as_u64().unwrap_or(12) as usize;
    let edits = args["max_experiments"].as_u64().unwrap_or(3) as usize;
    if !(1..=40).contains(&calls) || !(1..=10).contains(&edits) {
        return api_error(
            StatusCode::BAD_REQUEST,
            "Tool limit must be 1–40; experiment limit 1–10",
        );
    }
    if app.room.lock().unwrap().patient.is_null() {
        return api_error(StatusCode::BAD_REQUEST, "Load the patient first");
    }
    if let Err(e) = app.claim("operating") {
        return api_error(StatusCode::CONFLICT, e);
    }
    {
        let mut room = app.room.lock().unwrap();
        room.tool_calls = 0;
        room.experiments = 0;
        room.limits = json!({"max_tool_calls":calls,"max_experiments":edits});
    }
    tokio::spawn(async move {
        let result = app.operate(objective, calls, edits).await;
        app.finish(result).await;
    });
    Json(json!({"ok":true})).into_response()
}
async fn compare(State(app): State<App>, Json(_): Json<Value>) -> Response {
    if let Err(e) = app.claim("comparing") {
        return api_error(StatusCode::CONFLICT, e);
    }
    tokio::spawn(async move {
        let result = app
            .call("compare_patient", app.eval_args())
            .await
            .map(|_| ());
        app.finish(result).await;
    });
    Json(json!({"ok":true})).into_response()
}
async fn revert(State(app): State<App>, Json(_): Json<Value>) -> Response {
    {
        let mut room = app.room.lock().unwrap();
        if room.busy {
            app.cancel.store(true, Ordering::SeqCst);
            room.phase = "reverting".into();
            drop(room);
            app.event("system","Emergency revert requested. Cancelling Astra and waiting for the current bounded model action to stop.",json!({}));
            return Json(json!({"ok":true})).into_response();
        }
    }
    if let Err(e) = app.claim("reverting") {
        return api_error(StatusCode::CONFLICT, e);
    }
    tokio::spawn(async move {
        let result = app.call("revert_changes", json!({})).await.map(|_| ());
        app.finish(result).await;
    });
    Json(json!({"ok":true})).into_response()
}
async fn accept(State(app): State<App>, Json(_): Json<Value>) -> Response {
    if let Err(e) = app.claim("saving") {
        return api_error(StatusCode::CONFLICT, e);
    }
    tokio::spawn(async move {
        let result = app.call("save_patch", json!({})).await.map(|_| ());
        app.finish(result).await;
    });
    Json(json!({"ok":true})).into_response()
}
async fn apply_patch(State(app): State<App>, Json(args): Json<Value>) -> Response {
    if let Err(e) = app.claim("loading") {
        return api_error(StatusCode::CONFLICT, e);
    }
    tokio::spawn(async move {
        let result = app.call("apply_patch", args).await.map(|_| ());
        app.finish(result).await;
    });
    Json(json!({"ok":true})).into_response()
}
async fn inspect(State(app): State<App>, Json(args): Json<Value>) -> Response {
    if let Err(e) = app.claim("inspecting") {
        return api_error(StatusCode::CONFLICT, e);
    }
    let result = app.call("inspect_model", args).await;
    let outcome = result.as_ref().map(|_| ()).map_err(|e| anyhow!("{e:#}"));
    app.finish(outcome).await;
    match result {
        Ok(v) => Json(v).into_response(),
        Err(e) => api_error(StatusCode::BAD_REQUEST, e),
    }
}
async fn local_only(
    State(app): State<App>,
    headers: HeaderMap,
    request: Request,
    next: Next,
) -> Response {
    let host = headers
        .get(header::HOST)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    let allowed = [
        format!("127.0.0.1:{}", app.port),
        format!("localhost:{}", app.port),
    ];
    if !allowed.iter().any(|v| v == host) {
        return api_error(StatusCode::FORBIDDEN, "Local host required");
    }
    if let Some(origin) = headers.get(header::ORIGIN) {
        if origin.to_str().ok() != Some(&format!("http://{host}")) {
            return api_error(StatusCode::FORBIDDEN, "Same-origin requests only");
        }
    }
    let mut response = next.run(request).await;
    response
        .headers_mut()
        .insert(header::CACHE_CONTROL, "no-store".parse().unwrap());
    response
        .headers_mut()
        .insert(header::X_CONTENT_TYPE_OPTIONS, "nosniff".parse().unwrap());
    response
}

#[tokio::main]
async fn main() -> Result<()> {
    let _ = dotenvy::from_filename(".env.local");
    let _ = dotenvy::dotenv();
    let data = PathBuf::from(std::env::var("TABLE_DATA_DIR").unwrap_or_else(|_| "data".into()));
    std::fs::create_dir_all(&data)?;
    let app = App {
        patient: Arc::new(Mutex::new(None)),
        room: Arc::new(Mutex::new(Room {
            phase: "idle".into(),
            busy: false,
            patient: Value::Null,
            activity: vec![],
            comparison: Value::Null,
            error: None,
            active_tensor: None,
            tool_calls: 0,
            experiments: 0,
            limits: json!({"max_tool_calls":12,"max_experiments":3}),
            surgeon: json!({"model":"gpt-6-astra","transport":"Codex CLI"}),
        })),
        busy: Arc::new(AtomicBool::new(false)),
        cancel: Arc::new(AtomicBool::new(false)),
        data: Arc::new(std::fs::canonicalize(data)?),
        device: Arc::new(std::env::var("PATIENT_DEVICE").unwrap_or_else(|_| "auto".into())),
        port: std::env::var("TABLE_PORT")
            .unwrap_or_else(|_| "3000".into())
            .parse()
            .context("TABLE_PORT must be a valid port")?,
    };
    let router = Router::new()
        .route(
            "/",
            get(|| async { Html(include_str!("../static/index.html")) }),
        )
        .route(
            "/app.js",
            get(|| async {
                (
                    [(header::CONTENT_TYPE, "text/javascript; charset=utf-8")],
                    include_str!("../static/app.js"),
                )
            }),
        )
        .route(
            "/style.css",
            get(|| async {
                (
                    [(header::CONTENT_TYPE, "text/css; charset=utf-8")],
                    include_str!("../static/style.css"),
                )
            }),
        )
        .route("/api/state", get(state))
        .route("/api/load", post(load))
        .route("/api/surgery", post(surgery))
        .route("/api/compare", post(compare))
        .route("/api/revert", post(revert))
        .route("/api/accept", post(accept))
        .route("/api/inspect", post(inspect))
        .route("/api/patch", post(apply_patch))
        .layer(DefaultBodyLimit::max(32 * 1024))
        .layer(middleware::from_fn_with_state(app.clone(), local_only))
        .with_state(app.clone());
    let listener = tokio::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, app.port)).await?;
    println!(
        "The Operating Table · http://127.0.0.1:{} · surgeon: Codex / gpt-6-astra",
        app.port
    );
    axum::serve(listener, router).await?;
    Ok(())
}
