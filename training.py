import os
import re

from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

from pdt_txt import convert

app = Flask(__name__)


class Training:
    def __init__(self, file_name):
        self.doc_name = file_name.strip(".pdf")
        print(self.doc_name)
        self.domain = "model"
        if not os.path.exists('./input_file') and not os.path.exists('./output_model'):
            print("Created directory to store the model file")
            os.makedirs('./input_file')
            os.makedirs('./output_model')

        self.file_name = self.doc_name + ".txt"
        self.data_path = "./input_file/" + self.domain
        self.out_dir = "./output_model/"
        self.c = []

        self.statusContent = dict()
        self.statusContent["documentId"] = self.doc_name

        try:
            self.extractedContent = convert(self.doc_name + ".pdf")
            converted = open(self.file_name, "w", encoding='utf-8')
            converted.write(self.extractedContent)
            converted.close()
            print("writing content into file for training")
            print("Extracted the content for the training")

        except KeyError:
            self.statusContent["status"] = "FAILURE"
            self.statusContent["error"] = "No content found to train for the domain %s" % self.domain
            return

        if len(self.extractedContent.split()) <= 25:
            self.statusContent["status"] = "FAILURE"
            self.statusContent["error"] = "Not enough words Found to train"
            return

    def process(self, ):
        self.opening_file()
        status = self.model_creation()
        os.remove(self.file_name)
        os.remove(self.doc_name + ".pdf")
        return status

    def opening_file(self, ):
        print("Data is parsing for training")

        self.document = open(self.file_name, 'r', encoding='utf-8').readlines()

        for i in self.document:
            self.d = re.sub("[^a-zA-Z0-9.,$]", " ", i)
            self.c.append(re.sub(' +', ' ', self.d.strip()))
        self.conts = list(filter(None, list(set((' '.join(self.c)).split('.')))))

        self.f2 = open('./input_file/' + self.domain + '.txt', 'a')
        for i in range(len(self.conts)):
            self.f2.write("{0}".format(self.conts[i]))
            self.f2.write('\n')
        self.f2.close()

    def model_creation(self, ):
        print("Training Process has been started")
        status = {"domainName": self.domain, "documentName": self.doc_name, "status": "SUCCESS"}
        return status


@app.route('/', methods=['POST', 'GET'])
def train():
    if request.method == 'GET':
        return render_template('index2.html')
    elif request.method == 'POST':
        print("..............Model training starting..............")
        fname = request.files.get('title')
        fname.save('./' + secure_filename(fname.filename))
        response = Training(fname.filename).process()
        print(response)
        return render_template('index2.html')


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8001)
