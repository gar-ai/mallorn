//! Cryptographic signature support for secure OTA updates
//!
//! Provides ED25519 signing and verification for patch files,
//! enabling secure, authenticated model updates.

use crate::error::MallornError;
use crate::hash::sha256;
use ed25519_dalek::{
    Signature, Signer, SigningKey, Verifier, VerifyingKey, PUBLIC_KEY_LENGTH, SECRET_KEY_LENGTH,
    SIGNATURE_LENGTH,
};
use rand::rngs::OsRng;

/// A signed patch with embedded signature
#[derive(Debug, Clone)]
pub struct SignedPatch {
    /// The raw patch data
    pub patch_data: Vec<u8>,
    /// ED25519 signature over the patch data
    pub signature: [u8; SIGNATURE_LENGTH],
    /// Public key of the signer
    pub public_key: [u8; PUBLIC_KEY_LENGTH],
    /// Version for downgrade protection (monotonically increasing)
    pub version: u64,
}

impl SignedPatch {
    /// Create a new signed patch
    pub fn new(
        patch_data: Vec<u8>,
        signing_key: &SigningKey,
        version: u64,
    ) -> Result<Self, MallornError> {
        // Create message to sign: version || patch_hash
        let patch_hash = sha256(&patch_data);
        let mut message = version.to_le_bytes().to_vec();
        message.extend_from_slice(&patch_hash);

        // Sign the message
        let signature = signing_key.sign(&message);

        Ok(Self {
            patch_data,
            signature: signature.to_bytes(),
            public_key: signing_key.verifying_key().to_bytes(),
            version,
        })
    }

    /// Verify the signature and return the patch data if valid
    pub fn verify(&self) -> Result<&[u8], SignatureError> {
        let verifying_key = VerifyingKey::from_bytes(&self.public_key)
            .map_err(|_| SignatureError::InvalidPublicKey)?;

        // Reconstruct the signed message
        let patch_hash = sha256(&self.patch_data);
        let mut message = self.version.to_le_bytes().to_vec();
        message.extend_from_slice(&patch_hash);

        let signature =
            Signature::from_bytes(&self.signature);

        verifying_key
            .verify(&message, &signature)
            .map_err(|_| SignatureError::InvalidSignature)?;

        Ok(&self.patch_data)
    }

    /// Verify against a specific trusted public key
    pub fn verify_with_key(&self, trusted_key: &[u8; PUBLIC_KEY_LENGTH]) -> Result<&[u8], SignatureError> {
        // Check the public key matches
        if self.public_key != *trusted_key {
            return Err(SignatureError::UntrustedKey);
        }

        self.verify()
    }

    /// Check if version is greater than minimum (for downgrade protection)
    pub fn check_version(&self, minimum_version: u64) -> Result<(), SignatureError> {
        if self.version < minimum_version {
            return Err(SignatureError::DowngradeAttempt {
                patch_version: self.version,
                minimum_version,
            });
        }
        Ok(())
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // Magic: "MSIG" (Mallorn SIGnature)
        bytes.extend_from_slice(b"MSIG");

        // Version (8 bytes)
        bytes.extend_from_slice(&self.version.to_le_bytes());

        // Public key (32 bytes)
        bytes.extend_from_slice(&self.public_key);

        // Signature (64 bytes)
        bytes.extend_from_slice(&self.signature);

        // Patch data length (4 bytes)
        bytes.extend_from_slice(&(self.patch_data.len() as u32).to_le_bytes());

        // Patch data
        bytes.extend_from_slice(&self.patch_data);

        bytes
    }

    /// Deserialize from bytes
    pub fn from_bytes(data: &[u8]) -> Result<Self, SignatureError> {
        if data.len() < 112 {
            // 4 + 8 + 32 + 64 + 4 = 112 minimum
            return Err(SignatureError::InvalidFormat);
        }

        // Check magic
        if &data[0..4] != b"MSIG" {
            return Err(SignatureError::InvalidFormat);
        }

        // Parse version
        let version = u64::from_le_bytes(data[4..12].try_into().unwrap());

        // Parse public key
        let public_key: [u8; PUBLIC_KEY_LENGTH] = data[12..44].try_into().unwrap();

        // Parse signature
        let signature: [u8; SIGNATURE_LENGTH] = data[44..108].try_into().unwrap();

        // Parse patch data length
        let patch_len = u32::from_le_bytes(data[108..112].try_into().unwrap()) as usize;

        if data.len() < 112 + patch_len {
            return Err(SignatureError::InvalidFormat);
        }

        let patch_data = data[112..112 + patch_len].to_vec();

        Ok(Self {
            patch_data,
            signature,
            public_key,
            version,
        })
    }
}

/// Signature-related errors
#[derive(Debug, thiserror::Error)]
pub enum SignatureError {
    #[error("Invalid signature")]
    InvalidSignature,

    #[error("Invalid public key")]
    InvalidPublicKey,

    #[error("Untrusted key - signature key does not match trusted key")]
    UntrustedKey,

    #[error("Downgrade attempt: patch version {patch_version} < minimum {minimum_version}")]
    DowngradeAttempt {
        patch_version: u64,
        minimum_version: u64,
    },

    #[error("Invalid signed patch format")]
    InvalidFormat,
}

/// Generate a new ED25519 keypair
pub fn generate_keypair() -> (SigningKey, VerifyingKey) {
    let signing_key = SigningKey::generate(&mut OsRng);
    let verifying_key = signing_key.verifying_key();
    (signing_key, verifying_key)
}

/// Load a signing key from raw bytes
pub fn load_signing_key(bytes: &[u8; SECRET_KEY_LENGTH]) -> SigningKey {
    SigningKey::from_bytes(bytes)
}

/// Load a verifying key from raw bytes
pub fn load_verifying_key(bytes: &[u8; PUBLIC_KEY_LENGTH]) -> Result<VerifyingKey, SignatureError> {
    VerifyingKey::from_bytes(bytes).map_err(|_| SignatureError::InvalidPublicKey)
}

/// Check if data appears to be a signed patch (has MSIG magic)
pub fn is_signed_patch(data: &[u8]) -> bool {
    data.len() >= 4 && &data[0..4] == b"MSIG"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_keypair() {
        let (signing_key, verifying_key) = generate_keypair();
        assert_eq!(signing_key.verifying_key(), verifying_key);
    }

    #[test]
    fn test_sign_and_verify() {
        let (signing_key, _) = generate_keypair();
        let patch_data = b"test patch data".to_vec();

        let signed = SignedPatch::new(patch_data.clone(), &signing_key, 1).unwrap();
        let verified = signed.verify().unwrap();

        assert_eq!(verified, patch_data.as_slice());
    }

    #[test]
    fn test_verify_with_trusted_key() {
        let (signing_key, verifying_key) = generate_keypair();
        let patch_data = b"test patch data".to_vec();

        let signed = SignedPatch::new(patch_data.clone(), &signing_key, 1).unwrap();
        let result = signed.verify_with_key(&verifying_key.to_bytes());

        assert!(result.is_ok());
    }

    #[test]
    fn test_untrusted_key_rejected() {
        let (signing_key, _) = generate_keypair();
        let (_, other_verifying_key) = generate_keypair();
        let patch_data = b"test patch data".to_vec();

        let signed = SignedPatch::new(patch_data, &signing_key, 1).unwrap();
        let result = signed.verify_with_key(&other_verifying_key.to_bytes());

        assert!(matches!(result, Err(SignatureError::UntrustedKey)));
    }

    #[test]
    fn test_downgrade_protection() {
        let (signing_key, _) = generate_keypair();
        let patch_data = b"test patch data".to_vec();

        let signed = SignedPatch::new(patch_data, &signing_key, 5).unwrap();

        // Version 5 >= minimum 3: OK
        assert!(signed.check_version(3).is_ok());

        // Version 5 >= minimum 5: OK
        assert!(signed.check_version(5).is_ok());

        // Version 5 < minimum 10: Error
        assert!(matches!(
            signed.check_version(10),
            Err(SignatureError::DowngradeAttempt { .. })
        ));
    }

    #[test]
    fn test_serialize_deserialize() {
        let (signing_key, _) = generate_keypair();
        let patch_data = b"test patch data with more content".to_vec();

        let signed = SignedPatch::new(patch_data.clone(), &signing_key, 42).unwrap();
        let bytes = signed.to_bytes();
        let restored = SignedPatch::from_bytes(&bytes).unwrap();

        assert_eq!(restored.patch_data, patch_data);
        assert_eq!(restored.version, 42);
        assert_eq!(restored.public_key, signed.public_key);
        assert_eq!(restored.signature, signed.signature);

        // Verify the restored patch
        assert!(restored.verify().is_ok());
    }

    #[test]
    fn test_is_signed_patch() {
        assert!(is_signed_patch(b"MSIG...."));
        assert!(!is_signed_patch(b"TFLP...."));
        assert!(!is_signed_patch(b"MSI"));
    }

    #[test]
    fn test_tampered_data_rejected() {
        let (signing_key, _) = generate_keypair();
        let patch_data = b"original data".to_vec();

        let mut signed = SignedPatch::new(patch_data, &signing_key, 1).unwrap();

        // Tamper with the data
        signed.patch_data[0] = b'X';

        // Verification should fail
        assert!(matches!(signed.verify(), Err(SignatureError::InvalidSignature)));
    }
}
