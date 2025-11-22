"""
WSGI config for burn_calories project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.2/howto/deployment/wsgi/
"""
import os
from django.core.wsgi import get_wsgi_application

from download_data import ensure_files
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'burn_calories.settings')

ensure_files()

application = get_wsgi_application()