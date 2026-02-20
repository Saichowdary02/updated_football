import os
import requests
from jose import jwt
from flask import request, abort
from dotenv import load_dotenv
# Load environment variables
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

CLERK_ISSUER = os.getenv("CLERK_ISSUER")
if not CLERK_ISSUER:
    # Fallback or error
    print("WARNING: CLERK_ISSUER not set in .env")
    CLERK_ISSUER = "https://clerk.your-domain.clerk.accounts.dev"

CLERK_JWKS_URL = f"{CLERK_ISSUER}/.well-known/jwks.json"

# Fetch JWKS (JSON Web Key Set) from Clerk
try:
    jwks = requests.get(CLERK_JWKS_URL).json()
except Exception as e:
    print(f"Error fetching JWKS from {CLERK_JWKS_URL}: {e}")
    jwks = {}

def require_auth():
    """
    Verifies the Clerk JWT token from the Authorization header.
    Returns the decoded token payload if valid, otherwise aborts with 401.
    """
    auth_header = request.headers.get("Authorization")

    if not auth_header:
        abort(401, "Missing Authorization header")

    token = auth_header.replace("Bearer ", "")

    try:
        # Verify the token using the JWKS
        payload = jwt.decode(
            token,
            jwks,
            algorithms=["RS256"],
            # options={"verify_aud": False}, # define audience if needed, usually not strictly required for this simple setup if frontend is not specifying one
            audience=os.getenv("CLERK_FRONTEND_API", "https://through-hen-93.clerk.accounts.dev"), # Verify if your token has an audience (azp or aud)
            issuer=CLERK_ISSUER
        )
        return payload  # contains user_id, email, etc.
    except jwt.ExpiredSignatureError:
        abort(401, "Token has expired")
    except jwt.JWTClaimsError:
        abort(401, "Invalid claims. Please check the audience and issuer.")
    except Exception as e:
        print(f"Token verification failed: {e}")
        abort(401, "Invalid token")
