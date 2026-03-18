use crate::{AlignrustError, EngineConfig};
use crate::espeak::EspeakPhonemizer;
use libc::{c_int, pid_t};
use std::fs::File;
use std::io::{Read, Write};
use std::os::fd::FromRawFd;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard};

pub trait Phonemizer: Send + Sync {
    fn text_to_phonemes(&self, text: &str) -> Result<String, AlignrustError>;
}

impl Phonemizer for EspeakPhonemizer {
    fn text_to_phonemes(&self, text: &str) -> Result<String, AlignrustError> {
        EspeakPhonemizer::text_to_phonemes(self, text)
    }
}

struct WorkerIo {
    write: File,
    read: File,
}

struct Worker {
    pid: pid_t,
    io: Mutex<WorkerIo>,
}

pub struct ForkedEspeakPool {
    workers: Vec<Worker>,
    rr: AtomicUsize,
}

impl ForkedEspeakPool {
    pub fn new(cfg: &EngineConfig, workers: usize) -> Result<Self, AlignrustError> {
        if workers == 0 {
            return Err(AlignrustError::Message("workers must be > 0".to_owned()));
        }

        let mut out: Vec<Worker> = Vec::with_capacity(workers);
        for _ in 0..workers {
            out.push(spawn_worker(cfg)?);
        }

        Ok(Self {
            workers: out,
            rr: AtomicUsize::new(0),
        })
    }

    fn acquire(&self) -> Result<MutexGuard<'_, WorkerIo>, AlignrustError> {
        let start = self.rr.fetch_add(1, Ordering::Relaxed) % self.workers.len();
        for k in 0..self.workers.len() {
            let idx = (start + k) % self.workers.len();
            if let Ok(g) = self.workers[idx].io.try_lock() {
                return Ok(g);
            }
        }
        self.workers[start]
            .io
            .lock()
            .map_err(|_| AlignrustError::Message("worker io mutex poisoned".to_owned()))
    }
}

impl Phonemizer for ForkedEspeakPool {
    fn text_to_phonemes(&self, text: &str) -> Result<String, AlignrustError> {
        let mut io = self.acquire()?;

        let bytes = text.as_bytes();
        let len: u32 = bytes
            .len()
            .try_into()
            .map_err(|_| AlignrustError::Message("input too large".to_owned()))?;
        io.write.write_all(&len.to_le_bytes()).map_err(|e| {
            AlignrustError::Message(format!("worker write failed: {e}"))
        })?;
        io.write.write_all(bytes).map_err(|e| {
            AlignrustError::Message(format!("worker write failed: {e}"))
        })?;
        io.write.flush().map_err(|e| {
            AlignrustError::Message(format!("worker flush failed: {e}"))
        })?;

        let mut len_buf = [0u8; 4];
        io.read.read_exact(&mut len_buf).map_err(|e| {
            AlignrustError::Message(format!("worker read failed: {e}"))
        })?;
        let out_len = u32::from_le_bytes(len_buf) as usize;
        let mut out = vec![0u8; out_len];
        io.read.read_exact(&mut out).map_err(|e| {
            AlignrustError::Message(format!("worker read failed: {e}"))
        })?;
        String::from_utf8(out).map_err(|_| AlignrustError::Message("worker returned non-utf8".to_owned()))
    }
}

impl Drop for ForkedEspeakPool {
    fn drop(&mut self) {
        for w in &self.workers {
            let _ = unsafe { libc::kill(w.pid, libc::SIGTERM) };
        }
        for w in &self.workers {
            let mut status: c_int = 0;
            unsafe {
                libc::waitpid(w.pid, &mut status as *mut c_int, 0);
            }
        }
    }
}

fn spawn_worker(cfg: &EngineConfig) -> Result<Worker, AlignrustError> {
    let mut p2c = [0; 2];
    let mut c2p = [0; 2];

    let ok1 = unsafe { libc::pipe(p2c.as_mut_ptr()) };
    if ok1 != 0 {
        return Err(AlignrustError::Message("pipe failed".to_owned()));
    }
    let ok2 = unsafe { libc::pipe(c2p.as_mut_ptr()) };
    if ok2 != 0 {
        unsafe {
            libc::close(p2c[0]);
            libc::close(p2c[1]);
        }
        return Err(AlignrustError::Message("pipe failed".to_owned()));
    }

    let pid = unsafe { libc::fork() };
    if pid < 0 {
        unsafe {
            libc::close(p2c[0]);
            libc::close(p2c[1]);
            libc::close(c2p[0]);
            libc::close(c2p[1]);
        }
        return Err(AlignrustError::Message("fork failed".to_owned()));
    }

    if pid == 0 {
        unsafe {
            libc::close(p2c[1]);
            libc::close(c2p[0]);
        }

        let phon = match EspeakPhonemizer::new(cfg) {
            Ok(v) => v,
            Err(_) => unsafe { libc::_exit(1) },
        };

        let mut r = unsafe { File::from_raw_fd(p2c[0]) };
        let mut w = unsafe { File::from_raw_fd(c2p[1]) };

        let mut len_buf = [0u8; 4];
        loop {
            if r.read_exact(&mut len_buf).is_err() {
                break;
            }
            let n = u32::from_le_bytes(len_buf) as usize;
            let mut buf = vec![0u8; n];
            if r.read_exact(&mut buf).is_err() {
                break;
            }
            let text = match std::str::from_utf8(&buf) {
                Ok(s) => s,
                Err(_) => "",
            };
            let out = phon.text_to_phonemes(text).unwrap_or_default();
            let out_bytes = out.as_bytes();
            let out_len: u32 = match out_bytes.len().try_into() {
                Ok(v) => v,
                Err(_) => 0,
            };
            if w.write_all(&out_len.to_le_bytes()).is_err() {
                break;
            }
            if w.write_all(out_bytes).is_err() {
                break;
            }
            if w.flush().is_err() {
                break;
            }
        }
        unsafe { libc::_exit(0) }
    }

    unsafe {
        libc::close(p2c[0]);
        libc::close(c2p[1]);
    }

    let write = unsafe { File::from_raw_fd(p2c[1]) };
    let read = unsafe { File::from_raw_fd(c2p[0]) };

    Ok(Worker {
        pid,
        io: Mutex::new(WorkerIo { write, read }),
    })
}
