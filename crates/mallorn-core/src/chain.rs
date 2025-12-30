//! Delta chain operations
//!
//! This module provides serialization, deserialization, and manipulation
//! of patch chains for incremental model updates.

use crate::{crc32, ChainError, ChainLink, ChainMetadata, LinkMetadata, PatchChain};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{Cursor, Read, Write};

/// Magic bytes for chain files
pub const CHAIN_MAGIC: &[u8; 4] = b"MCHN";

/// Current chain format version
pub const CHAIN_VERSION: u32 = 1;

/// Serialize a patch chain to bytes
pub fn serialize_chain(chain: &PatchChain) -> Result<Vec<u8>, std::io::Error> {
    let mut buf = Vec::new();

    // Magic
    buf.write_all(CHAIN_MAGIC)?;

    // Version
    buf.write_u32::<LittleEndian>(chain.version)?;

    // Chain ID
    write_string(&mut buf, &chain.chain_id)?;

    // Metadata as JSON
    let metadata_json = serde_json::to_string(&chain.metadata)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    write_string(&mut buf, &metadata_json)?;

    // Number of links
    buf.write_u32::<LittleEndian>(chain.links.len() as u32)?;

    // Each link
    for link in &chain.links {
        serialize_link(&mut buf, link)?;
    }

    // CRC32 checksum
    let checksum = crc32(&buf);
    buf.write_u32::<LittleEndian>(checksum)?;

    Ok(buf)
}

/// Serialize a single chain link
fn serialize_link<W: Write>(writer: &mut W, link: &ChainLink) -> Result<(), std::io::Error> {
    // Index
    writer.write_u32::<LittleEndian>(link.index)?;

    // Hashes
    writer.write_all(&link.source_hash)?;
    writer.write_all(&link.target_hash)?;

    // Format
    write_string(writer, &link.format)?;

    // Link metadata as JSON
    let meta_json = serde_json::to_string(&link.link_metadata)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    write_string(writer, &meta_json)?;

    // Patch data
    writer.write_u32::<LittleEndian>(link.patch_data.len() as u32)?;
    writer.write_all(&link.patch_data)?;

    Ok(())
}

/// Write a length-prefixed string
fn write_string<W: Write>(writer: &mut W, s: &str) -> Result<(), std::io::Error> {
    let bytes = s.as_bytes();
    writer.write_u32::<LittleEndian>(bytes.len() as u32)?;
    writer.write_all(bytes)?;
    Ok(())
}

/// Deserialize a patch chain from bytes
pub fn deserialize_chain(data: &[u8]) -> Result<PatchChain, ChainError> {
    if data.len() < 8 {
        return Err(ChainError::InvalidChain("Data too small".into()));
    }

    // Verify magic
    if &data[0..4] != CHAIN_MAGIC {
        return Err(ChainError::InvalidChain("Invalid magic bytes".into()));
    }

    // Verify CRC32
    let payload = &data[..data.len() - 4];
    let stored_crc = u32::from_le_bytes([
        data[data.len() - 4],
        data[data.len() - 3],
        data[data.len() - 2],
        data[data.len() - 1],
    ]);
    let computed_crc = crc32(payload);
    if stored_crc != computed_crc {
        return Err(ChainError::InvalidChain("CRC32 checksum mismatch".into()));
    }

    let mut cursor = Cursor::new(payload);
    cursor.set_position(4); // Skip magic

    // Version
    let version = cursor
        .read_u32::<LittleEndian>()
        .map_err(|e| ChainError::InvalidChain(e.to_string()))?;

    if version != CHAIN_VERSION {
        return Err(ChainError::InvalidChain(format!(
            "Unsupported version: {}",
            version
        )));
    }

    // Chain ID
    let chain_id = read_string(&mut cursor)?;

    // Metadata
    let metadata_json = read_string(&mut cursor)?;
    let metadata: ChainMetadata = serde_json::from_str(&metadata_json)
        .map_err(|e| ChainError::InvalidChain(format!("Invalid metadata: {}", e)))?;

    // Number of links
    let num_links = cursor
        .read_u32::<LittleEndian>()
        .map_err(|e| ChainError::InvalidChain(e.to_string()))? as usize;

    // Read links
    let mut links = Vec::with_capacity(num_links);
    for _ in 0..num_links {
        let link = deserialize_link(&mut cursor)?;
        links.push(link);
    }

    Ok(PatchChain {
        version,
        chain_id,
        links,
        metadata,
    })
}

/// Deserialize a single chain link
fn deserialize_link(cursor: &mut Cursor<&[u8]>) -> Result<ChainLink, ChainError> {
    // Index
    let index = cursor
        .read_u32::<LittleEndian>()
        .map_err(|e| ChainError::InvalidChain(e.to_string()))?;

    // Hashes
    let mut source_hash = [0u8; 32];
    let mut target_hash = [0u8; 32];
    cursor
        .read_exact(&mut source_hash)
        .map_err(|e| ChainError::InvalidChain(e.to_string()))?;
    cursor
        .read_exact(&mut target_hash)
        .map_err(|e| ChainError::InvalidChain(e.to_string()))?;

    // Format
    let format = read_string(cursor)?;

    // Link metadata
    let meta_json = read_string(cursor)?;
    let link_metadata: LinkMetadata = serde_json::from_str(&meta_json)
        .map_err(|e| ChainError::InvalidChain(format!("Invalid link metadata: {}", e)))?;

    // Patch data
    let data_len = cursor
        .read_u32::<LittleEndian>()
        .map_err(|e| ChainError::InvalidChain(e.to_string()))? as usize;
    let mut patch_data = vec![0u8; data_len];
    cursor
        .read_exact(&mut patch_data)
        .map_err(|e| ChainError::InvalidChain(e.to_string()))?;

    Ok(ChainLink {
        index,
        source_hash,
        target_hash,
        patch_data,
        format,
        link_metadata,
    })
}

/// Read a length-prefixed string
fn read_string(cursor: &mut Cursor<&[u8]>) -> Result<String, ChainError> {
    let len = cursor
        .read_u32::<LittleEndian>()
        .map_err(|e| ChainError::InvalidChain(e.to_string()))? as usize;
    let mut bytes = vec![0u8; len];
    cursor
        .read_exact(&mut bytes)
        .map_err(|e| ChainError::InvalidChain(e.to_string()))?;
    String::from_utf8(bytes).map_err(|e| ChainError::InvalidChain(format!("Invalid UTF-8: {}", e)))
}

/// Check if data looks like a chain file
pub fn is_chain(data: &[u8]) -> bool {
    data.len() >= 4 && &data[0..4] == CHAIN_MAGIC
}

/// Get file extension for chain files
pub fn chain_extension() -> &'static str {
    "mchn"
}

// =============================================================================
// Chain Optimization Functions
// =============================================================================

/// Find the indices of links needed to update from source to target hash.
///
/// Returns `None` if no path exists, or `Some(vec![])` if source == target.
/// The returned indices are in order and can be used to apply patches sequentially.
pub fn find_update_path(chain: &PatchChain, from: &[u8; 32], to: &[u8; 32]) -> Option<Vec<usize>> {
    if from == to {
        return Some(vec![]);
    }

    // Find the starting link (source_hash matches 'from')
    let start_idx = chain.links.iter().position(|l| &l.source_hash == from)?;

    let mut path = Vec::new();
    let mut current_hash = *from;

    for (i, link) in chain.links.iter().enumerate().skip(start_idx) {
        if link.source_hash == current_hash {
            path.push(i);
            current_hash = link.target_hash;

            if &current_hash == to {
                return Some(path);
            }
        }
    }

    None
}

/// Calculate the total compressed size of applying a path of patches.
pub fn path_size(chain: &PatchChain, path: &[usize]) -> usize {
    path.iter()
        .filter_map(|&i| chain.links.get(i))
        .map(|link| link.patch_data.len())
        .sum()
}

/// Get statistics about a chain.
#[derive(Debug, Clone)]
pub struct ChainStats {
    /// Number of links in the chain
    pub num_links: usize,
    /// Total size of all patch data
    pub total_patch_size: usize,
    /// Average patch size
    pub avg_patch_size: usize,
    /// Largest patch size
    pub max_patch_size: usize,
    /// Smallest patch size
    pub min_patch_size: usize,
    /// List of version strings (if available)
    pub versions: Vec<String>,
}

/// Calculate statistics for a chain.
pub fn chain_stats(chain: &PatchChain) -> ChainStats {
    let sizes: Vec<usize> = chain.links.iter().map(|l| l.patch_data.len()).collect();
    let total: usize = sizes.iter().sum();

    let mut versions = Vec::new();
    if let Some(base_ver) = &chain.metadata.base_version {
        versions.push(base_ver.clone());
    }
    for link in &chain.links {
        if let Some(ver) = &link.link_metadata.target_version {
            if !versions.contains(ver) {
                versions.push(ver.clone());
            }
        }
    }

    ChainStats {
        num_links: chain.links.len(),
        total_patch_size: total,
        avg_patch_size: if chain.links.is_empty() {
            0
        } else {
            total / chain.links.len()
        },
        max_patch_size: sizes.iter().copied().max().unwrap_or(0),
        min_patch_size: sizes.iter().copied().min().unwrap_or(0),
        versions,
    }
}

/// Merge adjacent patches in a chain to create a more compact representation.
///
/// This is useful when you have a chain like v1→v2→v3→v4 and want to
/// create a single v1→v4 patch.
///
/// Note: This is a metadata-level merge. The actual patch squashing
/// requires format-specific logic and is done in format crates.
pub fn merge_chain_metadata(chain: &PatchChain, from_idx: usize, to_idx: usize) -> Option<ChainLink> {
    if from_idx > to_idx || to_idx >= chain.links.len() {
        return None;
    }

    let first_link = &chain.links[from_idx];
    let last_link = &chain.links[to_idx];

    // Collect all patch data (for later squashing by format-specific code)
    let combined_patch_data: Vec<u8> = chain.links[from_idx..=to_idx]
        .iter()
        .flat_map(|l| l.patch_data.iter().copied())
        .collect();
    let combined_size = combined_patch_data.len();

    Some(ChainLink {
        index: from_idx as u32,
        source_hash: first_link.source_hash,
        target_hash: last_link.target_hash,
        patch_data: combined_patch_data,
        format: first_link.format.clone(),
        link_metadata: LinkMetadata {
            source_version: first_link.link_metadata.source_version.clone(),
            target_version: last_link.link_metadata.target_version.clone(),
            patch_size: combined_size,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        },
    })
}

/// Create a new chain from a subset of links.
pub fn subchain(chain: &PatchChain, from_hash: &[u8; 32], to_hash: &[u8; 32]) -> Option<PatchChain> {
    let links = chain.links_between(from_hash, to_hash)?;

    if links.is_empty() {
        return None;
    }

    let mut new_chain = PatchChain::new(&format!("{}_subset", chain.chain_id));
    new_chain.metadata = ChainMetadata {
        base_version: links.first().and_then(|l| l.link_metadata.source_version.clone()),
        head_version: links.last().and_then(|l| l.link_metadata.target_version.clone()),
        total_links: links.len(),
        created_at: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
        description: Some(format!("Subset of {}", chain.chain_id)),
    };

    for (i, link) in links.into_iter().enumerate() {
        let new_link = ChainLink {
            index: i as u32,
            source_hash: link.source_hash,
            target_hash: link.target_hash,
            patch_data: link.patch_data.clone(),
            format: link.format.clone(),
            link_metadata: link.link_metadata.clone(),
        };
        let _ = new_chain.add_link(new_link);
    }

    Some(new_chain)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_chain() -> PatchChain {
        let mut chain = PatchChain::new("test_chain");

        let link1 = ChainLink {
            index: 0,
            source_hash: [0xAA; 32],
            target_hash: [0xBB; 32],
            patch_data: vec![1, 2, 3, 4, 5],
            format: "tflp".into(),
            link_metadata: LinkMetadata {
                source_version: Some("v1.0".into()),
                target_version: Some("v1.1".into()),
                patch_size: 5,
                created_at: 1234567890,
            },
        };

        let link2 = ChainLink {
            index: 1,
            source_hash: [0xBB; 32],
            target_hash: [0xCC; 32],
            patch_data: vec![6, 7, 8],
            format: "tflp".into(),
            link_metadata: LinkMetadata {
                source_version: Some("v1.1".into()),
                target_version: Some("v1.2".into()),
                patch_size: 3,
                created_at: 1234567900,
            },
        };

        chain.add_link(link1).unwrap();
        chain.add_link(link2).unwrap();
        chain
    }

    #[test]
    fn test_serialize_deserialize_roundtrip() {
        let chain = make_test_chain();
        let serialized = serialize_chain(&chain).unwrap();

        // Check magic
        assert_eq!(&serialized[0..4], CHAIN_MAGIC);

        // Deserialize
        let deserialized = deserialize_chain(&serialized).unwrap();

        // Verify
        assert_eq!(deserialized.chain_id, chain.chain_id);
        assert_eq!(deserialized.links.len(), chain.links.len());
        assert_eq!(deserialized.links[0].source_hash, chain.links[0].source_hash);
        assert_eq!(deserialized.links[1].target_hash, chain.links[1].target_hash);
    }

    #[test]
    fn test_is_chain() {
        assert!(is_chain(b"MCHN1234"));
        assert!(!is_chain(b"TFLP1234"));
        assert!(!is_chain(b"MCH")); // Too short
    }

    #[test]
    fn test_chain_extension() {
        assert_eq!(chain_extension(), "mchn");
    }

    #[test]
    fn test_chain_add_link_validates_hashes() {
        let mut chain = PatchChain::new("test");

        let link1 = ChainLink {
            index: 0,
            source_hash: [0xAA; 32],
            target_hash: [0xBB; 32],
            patch_data: vec![1, 2, 3],
            format: "test".into(),
            link_metadata: LinkMetadata::default(),
        };

        chain.add_link(link1).unwrap();

        // This link doesn't connect (source != previous target)
        let bad_link = ChainLink {
            index: 1,
            source_hash: [0xCC; 32], // Wrong! Should be 0xBB
            target_hash: [0xDD; 32],
            patch_data: vec![4, 5, 6],
            format: "test".into(),
            link_metadata: LinkMetadata::default(),
        };

        let result = chain.add_link(bad_link);
        assert!(matches!(result, Err(ChainError::HashMismatch { .. })));
    }

    #[test]
    fn test_chain_base_and_head_hash() {
        let chain = make_test_chain();

        assert_eq!(chain.base_hash(), Some([0xAA; 32]));
        assert_eq!(chain.head_hash(), Some([0xCC; 32]));
    }

    #[test]
    fn test_chain_contains_hash() {
        let chain = make_test_chain();

        assert!(chain.contains_hash(&[0xAA; 32]));
        assert!(chain.contains_hash(&[0xBB; 32]));
        assert!(chain.contains_hash(&[0xCC; 32]));
        assert!(!chain.contains_hash(&[0xDD; 32]));
    }

    #[test]
    fn test_chain_links_between() {
        let chain = make_test_chain();

        // Full chain
        let links = chain.links_between(&[0xAA; 32], &[0xCC; 32]).unwrap();
        assert_eq!(links.len(), 2);

        // First link only
        let links = chain.links_between(&[0xAA; 32], &[0xBB; 32]).unwrap();
        assert_eq!(links.len(), 1);

        // Not in chain
        let links = chain.links_between(&[0xDD; 32], &[0xEE; 32]);
        assert!(links.is_none());
    }

    #[test]
    fn test_chain_total_size() {
        let chain = make_test_chain();
        assert_eq!(chain.total_patch_size(), 8); // 5 + 3
    }

    #[test]
    fn test_empty_chain() {
        let chain = PatchChain::new("empty");
        assert_eq!(chain.base_hash(), None);
        assert_eq!(chain.head_hash(), None);
        assert_eq!(chain.total_patch_size(), 0);
    }

    #[test]
    fn test_find_update_path() {
        let chain = make_test_chain();

        // Full path from AA to CC
        let path = find_update_path(&chain, &[0xAA; 32], &[0xCC; 32]).unwrap();
        assert_eq!(path, vec![0, 1]);

        // First link only
        let path = find_update_path(&chain, &[0xAA; 32], &[0xBB; 32]).unwrap();
        assert_eq!(path, vec![0]);

        // Second link only
        let path = find_update_path(&chain, &[0xBB; 32], &[0xCC; 32]).unwrap();
        assert_eq!(path, vec![1]);

        // Same hash - empty path
        let path = find_update_path(&chain, &[0xAA; 32], &[0xAA; 32]).unwrap();
        assert!(path.is_empty());

        // Non-existent path
        let path = find_update_path(&chain, &[0xDD; 32], &[0xEE; 32]);
        assert!(path.is_none());
    }

    #[test]
    fn test_path_size() {
        let chain = make_test_chain();

        // Full path
        let path = find_update_path(&chain, &[0xAA; 32], &[0xCC; 32]).unwrap();
        assert_eq!(path_size(&chain, &path), 8); // 5 + 3

        // First link only
        let path = find_update_path(&chain, &[0xAA; 32], &[0xBB; 32]).unwrap();
        assert_eq!(path_size(&chain, &path), 5);

        // Empty path
        assert_eq!(path_size(&chain, &[]), 0);
    }

    #[test]
    fn test_chain_stats() {
        let chain = make_test_chain();
        let stats = chain_stats(&chain);

        assert_eq!(stats.num_links, 2);
        assert_eq!(stats.total_patch_size, 8);
        assert_eq!(stats.avg_patch_size, 4);
        assert_eq!(stats.max_patch_size, 5);
        assert_eq!(stats.min_patch_size, 3);
        assert_eq!(stats.versions.len(), 3); // v1.0, v1.1, v1.2
    }

    #[test]
    fn test_merge_chain_metadata() {
        let chain = make_test_chain();

        // Merge first two links
        let merged = merge_chain_metadata(&chain, 0, 1).unwrap();

        assert_eq!(merged.source_hash, [0xAA; 32]);
        assert_eq!(merged.target_hash, [0xCC; 32]);
        assert_eq!(merged.patch_data.len(), 8); // Combined data
        assert_eq!(merged.link_metadata.source_version, Some("v1.0".into()));
        assert_eq!(merged.link_metadata.target_version, Some("v1.2".into()));

        // Invalid range
        assert!(merge_chain_metadata(&chain, 1, 0).is_none());
        assert!(merge_chain_metadata(&chain, 0, 10).is_none());
    }

    #[test]
    fn test_subchain() {
        let chain = make_test_chain();

        // Get full subchain
        let sub = subchain(&chain, &[0xAA; 32], &[0xCC; 32]).unwrap();
        assert_eq!(sub.links.len(), 2);
        assert!(sub.chain_id.contains("subset"));

        // Get partial subchain
        let sub = subchain(&chain, &[0xAA; 32], &[0xBB; 32]).unwrap();
        assert_eq!(sub.links.len(), 1);

        // Non-existent range
        assert!(subchain(&chain, &[0xDD; 32], &[0xEE; 32]).is_none());
    }
}
