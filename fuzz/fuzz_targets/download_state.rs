#![no_main]

use libfuzzer_sys::fuzz_target;
use mallorn_core::network::DownloadState;
use std::io::Cursor;

fuzz_target!(|data: &[u8]| {
    // Fuzz DownloadState JSON parsing
    // Should never panic, even with arbitrary input

    // Try to parse as JSON (DownloadState uses serde_json)
    if let Ok(s) = std::str::from_utf8(data) {
        let _ = serde_json::from_str::<DownloadState>(s);
    }

    // Also try parsing the bytes directly
    let _ = serde_json::from_slice::<DownloadState>(data);

    // Test with valid-looking but corrupted data
    if data.len() >= 32 {
        let url = String::from_utf8_lossy(&data[0..16.min(data.len())]).to_string();
        let downloaded = if data.len() >= 8 {
            u64::from_le_bytes(data[0..8].try_into().unwrap_or([0; 8]))
        } else {
            0
        };
        let total_size = if data.len() >= 16 {
            u64::from_le_bytes(data[8..16].try_into().unwrap_or([0; 8]))
        } else {
            1000
        };

        let state = DownloadState {
            url,
            downloaded,
            total_size,
            etag: None,
            partial_path: "/tmp/test".to_string(),
            target_path: "/tmp/out".to_string(),
        };

        // Test state methods
        let _ = state.progress();
        let _ = state.is_complete();

        // Roundtrip through JSON
        if let Ok(json) = serde_json::to_string(&state) {
            let _ = serde_json::from_str::<DownloadState>(&json);
        }
    }
});
