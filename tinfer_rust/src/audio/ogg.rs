use super::{AudioError, Result};

pub(crate) struct OggStream {
    serial: u32,
    sequence: u32,
}

impl OggStream {
    pub(crate) fn new(serial: u32) -> Self {
        assert_ne!(serial, 0, "Ogg stream serial must be nonzero");
        Self { serial, sequence: 0 }
    }

    pub(crate) fn packet(&mut self, packet: &[u8], granule: u64, flags: u8) -> Result<Vec<u8>> {
        let full_segments = packet.len() / 255;
        let segments = full_segments + 1;
        let segment_count = u8::try_from(segments).map_err(|_| AudioError::AudioTooLong)?;
        let mut page = Vec::with_capacity(27 + segments + packet.len());
        page.extend_from_slice(b"OggS");
        page.push(0);
        page.push(flags);
        page.extend_from_slice(&granule.to_le_bytes());
        page.extend_from_slice(&self.serial.to_le_bytes());
        page.extend_from_slice(&self.sequence.to_le_bytes());
        page.extend_from_slice(&0_u32.to_le_bytes());
        page.push(segment_count);
        page.extend(std::iter::repeat_n(255, full_segments));
        page.push((packet.len() % 255) as u8);
        page.extend_from_slice(packet);
        let checksum = crc(&page);
        page[22..26].copy_from_slice(&checksum.to_le_bytes());
        self.sequence += 1;
        Ok(page)
    }
}

pub(crate) fn crc(bytes: &[u8]) -> u32 {
    let mut checksum = 0_u32;
    for byte in bytes {
        checksum ^= u32::from(*byte) << 24;
        for _ in 0..8 {
            checksum = if checksum & 0x8000_0000 != 0 { (checksum << 1) ^ 0x04c1_1db7 } else { checksum << 1 };
        }
    }
    checksum
}
