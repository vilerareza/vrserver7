"""
ASGI config for vrserver project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/3.2/howto/deployment/asgi/
"""

import os
from channels.auth import AuthMiddlewareStack
from channels.routing import ProtocolTypeRouter, URLRouter
from django.core.asgi import get_asgi_application
from channels.security.websocket import AllowedHostsOriginValidator
import stream.routing 

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'vrserver.settings')

# Following line is adjusted based on Channels documentation
#application = get_asgi_application()
django_asgi_app = get_asgi_application()

application = ProtocolTypeRouter({
    "http": django_asgi_app,
    "websocket": 
            AuthMiddlewareStack(
            URLRouter(
                stream.routing.websocket_urlpatterns
            )
    )
})
    # Just HTTP for now. (We can add other protocols later.)
