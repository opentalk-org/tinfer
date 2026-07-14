use cxx::UniquePtr;

use crate::{Error, Result};

pub(crate) struct Handle(UniquePtr<ffi::Model>);

unsafe impl Send for Handle {}
unsafe impl Sync for Handle {}

impl Handle {
    pub fn stub() -> Result<Self> {
        ffi::load_stub().map(Self).map_err(|error| Error::Inference(error.to_string()))
    }

    pub fn styletts2(root: &str, architecture: &str, backend: u8, device: i32) -> Result<Self> {
        ffi::load_styletts2(root, architecture, backend, device).map(Self).map_err(|error| Error::Inference(error.to_string()))
    }

    pub fn generate(&self, tensors: Vec<ffi::Tensor>) -> Result<Vec<ffi::Tensor>> {
        self.0
            .as_ref()
            .expect("native model is loaded")
            .generate_batch(&ffi::Batch { tensors })
            .map(|output| output.tensors)
            .map_err(|error| Error::Inference(error.to_string()))
    }
}

pub(crate) fn tensor(name: &str, dtype: ffi::DType, shape: Vec<i64>, data: Vec<u8>) -> ffi::Tensor {
    ffi::Tensor { name: name.into(), dtype, shape, data }
}

#[cxx::bridge(namespace = "tinfer::native")]
pub(crate) mod ffi {
    #[repr(u8)]
    enum DType {
        F16 = 0,
        F32 = 1,
        I32 = 2,
        I64 = 3,
        Bool = 4,
    }

    struct Tensor {
        name: String,
        dtype: DType,
        shape: Vec<i64>,
        data: Vec<u8>,
    }

    struct Batch {
        tensors: Vec<Tensor>,
    }

    struct Output {
        tensors: Vec<Tensor>,
    }

    unsafe extern "C++" {
        include!("tinfer_rust/src/models/base/cpp/model.hpp");
        type Model;
        fn load_stub() -> Result<UniquePtr<Model>>;
        fn load_styletts2(root: &str, architecture: &str, backend: u8, device: i32) -> Result<UniquePtr<Model>>;
        fn generate_batch(self: &Model, batch: &Batch) -> Result<Output>;
        #[allow(dead_code)]
        fn cpu_duration_prefix(durations: &[f32], lengths: &[i32], speeds: &[f32], batch: i32, tokens: i32) -> Result<Vec<i32>>;
    }
}

#[cfg(test)]
mod tests {
    use super::ffi;

    #[test]
    fn cpu_duration_glue_uses_runtime_lengths_and_speed() {
        let result = ffi::cpu_duration_prefix(&[1.2, 2.6, 9.0, 0.2, 1.6, 2.4], &[2, 3], &[1.0, 2.0], 2, 3).unwrap();
        assert_eq!(result, vec![1, 3, 0, 1, 1, 1]);
    }
}
