from django.shortcuts import render

# Create your views here.
import json
from django.http import HttpResponse


def hello(request):
    resp = {'message': "success", 'result': 'ok'}
    resp['message'] = 'aaa'
    resp['result'] = 'ok'
    resp['data'] = "aaa"
    return HttpResponse(json.dumps(resp), content_type="application/json")