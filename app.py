from flask import Flask, render_template, request, make_response, jsonify
from static.api.tools import getFakeMIPScatteringMap

app = Flask(__name__)
# app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

@app.route('/')
def homepage():
    return render_template('index.html')

@app.route('/get-fake-mip-scattering', methods=['POST'])
def getFakeMip():
    request_data = request.get_json()
   
    laorao = request_data['LAORAO']
    caucra = request_data['CAUCRA']

    print("Get POST: request fake MIP for laorao ", laorao, " and caucra ", caucra)

    # Compute the map
    data, sizex, sizey, dtype = getFakeMIPScatteringMap()

    msg = {
        "status": "Done",
        "sizex": sizex,
        "sizey": sizey,
        "dtype": dtype,
        "data": data,
    }

    response = make_response(jsonify(msg))

    return response

# Example
@app.route('/json-example', methods=['POST'])
def json_example():

    request_data = request.get_json()
   
    laorao = request_data['LAORAO']
    caucra = request_data['CAUCRA']
    print("Get POST", laorao, caucra)

    msg = {
        "msg": "Working"
    }

    response = make_response(jsonify(msg))

    return response

if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)

