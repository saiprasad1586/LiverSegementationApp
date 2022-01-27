from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
from process_nifti import Predict

app = Flask(__name__)
db = SQLAlchemy(app)

from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:sai@localhost/liversgementationapi'
db = SQLAlchemy(app)
migrate = Migrate(app, db)


class Patient(db.Model):
    __tablename__ = 'patient'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String())
    age = db.Column(db.String())

    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __repr__(self):
        return '<id {}>'.format(self.id)


class patient_slice(db.Model):
    __tablename__ = 'patient_slice'
    __table_args__ = {'extend_existing': True}

    id = db.Column(db.Integer, primary_key=True)
    id_patient = db.Column(db.Integer, db.ForeignKey('patient.id'))
    slice_location = db.Column(db.String())

    def __init__(self, id_patient, slice_location):
        self.id_patient = id_patient
        self.slice_location = slice_location

    def __repr__(self):
        return '<id {}>'.format(self.id)


class User(db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    user_name = db.Column(db.String())
    password = db.Column(db.String())

    def __init__(self, user_name, password):
        self.user_name = user_name
        self.password = password

    def __repr__(self):
        return '<id {}>'.format(self.id)


class patient_slice_result(db.Model):
    __tablename__ = 'patient_slice_result'
    __table_args__ = {'extend_existing': True}

    id = db.Column(db.Integer, primary_key=True)
    id_patient = db.Column(db.Integer, db.ForeignKey('patient.id'))
    result_loction = db.Column(db.String())

    def __init__(self, id_patient, result_loction):
        self.id_patient = id_patient
        self.result_loction = result_loction

    def __repr__(self):
        return '<id {}>'.format(self.id)


@app.route('/')
def main_app():
    return render_template('index.html')


def commit_data(name, age, f):
    print(name, age)
    entry = Patient(name, age)
    db.session.add(entry)
    db.session.commit()
    upload_path = 'static/Uploads/' + secure_filename(f.filename)
    entry1 = patient_slice(entry.id, upload_path)
    db.session.add(entry1)
    f.save('static/Uploads/' + secure_filename(f.filename))
    path = predict_on_patient(entry.id, upload_path)
    entry2 = patient_slice_result(entry.id, result_loction=path)
    db.session.add(entry2)
    db.session.commit()
    return path, entry.id


def predict_on_patient(id, upload_path):
    print(upload_path)
    patient = Predict(id, upload_path)
    path = patient.return_final_path()

    return path


@app.route('/uploader', methods=['GET', 'POST'])
def uploader():
    if request.method == 'POST':
        name = request.form['name']
        age = request.form['age']
        f = request.files['file']
        path, id = commit_data(name, age, f)
        lst = [int(x) for x in range(64)]

        return render_template('results.html', folder=lst, id=id)


@app.route('/admin')
def admin():
    return render_template('admin.html')


if __name__ == '__main__':
    app.run(debug=True)
