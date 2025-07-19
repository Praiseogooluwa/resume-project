from .main import app

# This is required for Vercel
def handler(event, context):
    from mangum import Mangum
    asgi_handler = Mangum(app)
    return asgi_handler(event, context)
