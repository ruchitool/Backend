from flask import Flask
from flask_restful import Resource, Api, reqparse
from service import predict_req
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
api = Api(app)


class Predict(Resource):
    def post(self):
        parser = reqparse.RequestParser()  # initialize
        
        parser.add_argument('data', required=True)  # add args
        
        args = parser.parse_args()
        data = args['data']
        return predict_req(data)
    
api.add_resource(Predict, '/predict')
if __name__ == '__main__':
    app.run()