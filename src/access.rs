use anyhow::{bail, Result};
use axum::http::{header, HeaderMap};

#[derive(Clone, Debug)]
pub struct Access {
    port: u16,
    public_authority: Option<String>,
}

// Accept only DNS hostnames and an optional numeric port. Canonicalize the
// default HTTPS port so browser Origin serialization and proxy Host agree.
fn https_authority(value: &str) -> Option<String> {
    let (host, port) = match value.split_once(':') {
        Some((host, port)) => (host, Some(port.parse::<u16>().ok().filter(|p| *p != 0)?)),
        None => (value, None),
    };
    if host.is_empty()
        || host.len() > 253
        || !host.is_ascii()
        || host.split('.').any(|label| {
            label.is_empty()
                || label.len() > 63
                || label.starts_with('-')
                || label.ends_with('-')
                || !label
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'-')
        })
    {
        return None;
    }
    let host = host.to_ascii_lowercase();
    match port {
        Some(port) if port != 443 => Some(format!("{host}:{port}")),
        _ => Some(host),
    }
}

impl Access {
    pub fn new(port: u16, public_origin: Option<&str>) -> Result<Self> {
        let public_authority = match public_origin {
            Some(origin) => {
                let Some(authority) = origin.strip_prefix("https://").and_then(https_authority)
                else {
                    bail!("TABLE_PUBLIC_ORIGIN must be an exact HTTPS origin with a valid hostname and optional port");
                };
                Some(authority)
            }
            None => None,
        };
        Ok(Self {
            port,
            public_authority,
        })
    }

    pub fn allows(&self, headers: &HeaderMap) -> bool {
        let mut hosts = headers.get_all(header::HOST).iter();
        let Some(host) = hosts.next().and_then(|v| v.to_str().ok()) else {
            return false;
        };
        if hosts.next().is_some() {
            return false;
        }
        let local = host == format!("127.0.0.1:{}", self.port)
            || host == format!("localhost:{}", self.port);
        let public = self
            .public_authority
            .as_ref()
            .is_some_and(|expected| https_authority(host).as_ref() == Some(expected));
        if !local && !public {
            return false;
        }
        let mut origins = headers.get_all(header::ORIGIN).iter();
        let Some(origin) = origins.next() else {
            return true;
        };
        if origins.next().is_some() {
            return false;
        }
        let Ok(origin) = origin.to_str() else {
            return false;
        };
        if local && origin == format!("http://{host}") {
            return true;
        }
        public && origin.strip_prefix("https://").and_then(https_authority) == self.public_authority
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn headers(host: &str, origin: Option<&str>) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(header::HOST, host.parse().unwrap());
        if let Some(origin) = origin {
            headers.insert(header::ORIGIN, origin.parse().unwrap());
        }
        headers
    }

    #[test]
    fn public_origin_is_same_host_and_https() {
        let access = Access::new(3000, Some("https://operation-room.dimensionlab.net")).unwrap();
        assert!(access.allows(&headers(
            "operation-room.dimensionlab.net",
            Some("https://operation-room.dimensionlab.net")
        )));
        assert!(access.allows(&headers(
            "operation-room.dimensionlab.net:443",
            Some("https://operation-room.dimensionlab.net")
        )));
        assert!(!access.allows(&headers(
            "operation-room.dimensionlab.net",
            Some("https://evil.example")
        )));
        assert!(!access.allows(&headers(
            "operation-room.dimensionlab.net",
            Some("http://operation-room.dimensionlab.net")
        )));
        assert!(!access.allows(&headers(
            "evil.example",
            Some("https://operation-room.dimensionlab.net")
        )));
        assert!(!access.allows(&headers(
            "localhost:3000",
            Some("https://operation-room.dimensionlab.net")
        )));
    }

    #[test]
    fn local_access_remains_available_without_public_origin() {
        let access = Access::new(3000, None).unwrap();
        assert!(access.allows(&headers("localhost:3000", Some("http://localhost:3000"))));
        assert!(access.allows(&headers("127.0.0.1:3000", None)));
        assert!(!access.allows(&headers("127.0.0.1:3000", Some("http://localhost:3000"))));
        assert!(!access.allows(&headers("operation-room.dimensionlab.net", None)));
    }

    #[test]
    fn malformed_public_origin_is_rejected() {
        for origin in [
            "",
            "http://example.com",
            "https://example.com/",
            "https://example.com/path",
            "https://example.com?x=1",
            "https://example.com#x",
            "https://user@example.com",
            "https://bad_host",
            "https://example.com:0",
            "https://example.com:abc",
            "https://-bad.example",
        ] {
            assert!(
                Access::new(3000, Some(origin)).is_err(),
                "accepted {origin}"
            );
        }
    }
}
