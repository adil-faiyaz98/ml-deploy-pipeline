# homomorphic_encrypt_model.py
from phe import paillier

public_key, private_key = paillier.generate_paillier_keypair()

with open("keys/public.key", "w") as f:
    f.write(str(public_key))
with open("keys/private.key", "w") as f:
    f.write(str(private_key))

print("Model encryption keys generated.")