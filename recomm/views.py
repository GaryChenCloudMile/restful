import pandas as pd, restful_env as env

from datetime import datetime
from rest_framework.response import Response
from rest_framework import viewsets
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view

from .trainer import env
from .trainer.ctrl import Ctrl

class ViewsRecomm(viewsets.ModelViewSet):
    instance = None
    logger = env.logger('ViewsRecomm')

    @csrf_exempt
    def entry(self, req, method):
        ret = {}
        s = datetime.now()
        try:
            params = pd.Series(dict(req.data)).map(lambda e: e[0] if len(e) == 1 else e)
            self.logger.info('params: {}'.format(params))

            ret['response'] = getattr(Ctrl.instance, method)(params)
            ret[env.ERR_CDE] = '00'
        except Exception as e:
            ret[env.ERR_CDE] = '99'
            ret[env.ERR_MSG] = str(e)
            self.logger.error(e, exc_info=True)
        finally:
            self.logger.info('processing {} take time {}'.format(method, datetime.now() - s))

        return Response(ret)

ViewsRecomm.instance = ViewsRecomm()

# @api_view()
# @csrf_exempt
# def entry(req, method):
#     ret = {}
#     logger = Ctrl.instance.logger
#     s = datetime.now()
#     try:
#         params = pd.Series(dict(req.POST)).map(lambda e: e[0] if len(e) == 1 else e)
#         ret['response'] = getattr(Ctrl.instance, method)(params)
#         ret[env.ERR_CDE] = '00'
#     except Exception as e:
#         ret[env.ERR_CDE] = '99'
#         ret[env.ERR_MSG] = str(e)
#         logger.error(e, exc_info=True)
#     finally:
#         logger.info('processing {} take time {}'.format(method, datetime.now() - s))
#
#     return Response(ret)

