import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://facerecognition-84e51-default-rtdb.firebaseio.com/"
})

ref = db.reference('Students')

data = {
    "2021-0522-ST-0":{
        "name": "Kurt Palomo",
        "Program": "IT",
        "Year": "2nd Year",
        "Gender":"Male"
    },
    "2021-0262-ST-0":{
        "name": "Thea Jamasali",
        "Program": "IT",
        "Year": "2nd Year",
        "Gender":"Female"
    }

}

for key, value in data.items():
    ref.child(key).set(value)