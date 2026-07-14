use std::borrow::Cow;

use rubato::{FftFixedInOut, Resampler};

use super::{AudioError, Result};

const CHUNK_FRAMES: usize = 1_024;

pub(crate) struct StreamResampler {
    source_rate: u32,
    target_rate: u32,
    inner: Option<FftFixedInOut<f32>>,
    pending: Vec<f32>,
    delay_remaining: usize,
    input_frames: u64,
    output_frames: u64,
}

impl StreamResampler {
    pub(crate) fn new(source_rate: u32, target_rate: u32) -> Result<Self> {
        let inner = if source_rate == target_rate {
            None
        } else {
            Some(
                FftFixedInOut::new(source_rate as usize, target_rate as usize, CHUNK_FRAMES, 1)
                    .map_err(|error| AudioError::Resample(error.to_string()))?,
            )
        };
        let delay_remaining = inner.as_ref().map_or(0, Resampler::output_delay);
        Ok(Self { source_rate, target_rate, inner, pending: Vec::new(), delay_remaining, input_frames: 0, output_frames: 0 })
    }

    pub(crate) fn push<'a>(&mut self, samples: &'a [f32]) -> Result<Cow<'a, [f32]>> {
        let input_frames = self.input_frames + samples.len() as u64;
        let Some(resampler) = self.inner.as_mut() else {
            self.input_frames = input_frames;
            self.output_frames = input_frames;
            return Ok(Cow::Borrowed(samples));
        };
        self.input_frames = input_frames;
        self.pending.extend(samples.iter().map(|sample| sample.clamp(-1.0, 1.0)));
        let mut output = Vec::new();
        while self.pending.len() >= resampler.input_frames_next() {
            let frames = resampler.input_frames_next();
            let channels = [&self.pending[..frames]];
            let block = resampler.process(&channels, None).map_err(|error| AudioError::Resample(error.to_string()))?;
            self.pending.drain(..frames);
            append_delayed(&block[0], &mut output, &mut self.delay_remaining, &mut self.output_frames);
        }
        Ok(Cow::Owned(output))
    }

    pub(crate) fn finish(&mut self) -> Result<Vec<f32>> {
        let Some(resampler) = self.inner.as_mut() else {
            return Ok(Vec::new());
        };
        if self.input_frames == 0 {
            return Ok(Vec::new());
        }
        let block = if self.pending.is_empty() {
            resampler.process_partial::<&[f32]>(None, None)
        } else {
            let channels = [&self.pending[..]];
            resampler.process_partial(Some(&channels), None)
        }
        .map_err(|error| AudioError::Resample(error.to_string()))?;
        self.pending.clear();
        let mut output = Vec::new();
        append_delayed(&block[0], &mut output, &mut self.delay_remaining, &mut self.output_frames);
        let drained = resampler.process_partial::<&[f32]>(None, None).map_err(|error| AudioError::Resample(error.to_string()))?;
        append_delayed(&drained[0], &mut output, &mut self.delay_remaining, &mut self.output_frames);
        let expected = (self.input_frames * u64::from(self.target_rate)).div_ceil(u64::from(self.source_rate));
        let excess = self.output_frames.saturating_sub(expected) as usize;
        output.truncate(output.len().saturating_sub(excess));
        self.output_frames -= excess as u64;
        Ok(output)
    }
}

fn append_delayed(block: &[f32], output: &mut Vec<f32>, delay_remaining: &mut usize, output_frames: &mut u64) {
    let skipped = (*delay_remaining).min(block.len());
    *delay_remaining -= skipped;
    output.extend_from_slice(&block[skipped..]);
    *output_frames += (block.len() - skipped) as u64;
}
