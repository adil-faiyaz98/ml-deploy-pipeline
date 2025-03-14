# secure_inference.py
from phe import paillier
import json

with open("keys/public.key", "r") as f:
    public_key = paillier.PaillierPublicKey(int(f.read()))

encrypted_input = public_key.encrypt(42)
print(f"Encrypted inference input: {json.dumps(str(encrypted_input))}")