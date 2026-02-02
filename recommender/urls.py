from django.urls import path
from .views import recommend_movies, home

urlpatterns = [
    path('', home),
    path('recommend/', recommend_movies),
]
