from django.shortcuts import render

# Create your views here.
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .engine import recommend


@api_view(['GET'])
def recommend_movies(request):
    title = request.GET.get('movie')

    if not title:
        return Response({"error": "Please provide movie name"}, status=400)

    results = recommend(title)
    return Response(results)

def home(request):
    return render(request, 'index.html')