from flask import Flask, request, render_template

from Model import QA

app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def prediction():
    if request.method == 'GET':
        return render_template('index.html')
    elif request.method == 'POST':
        question = request.form.get('query')
        response = qa.predict(paragraph, question)
        answer = 'Sorry, I did not get that. Please rephrase your question'
        final_answer = None

        if response['confidence'] >= 0.1:
            final_answer = response['answer']

        if final_answer:
            print(final_answer)
            kwargs = {
                'query': question,
                'answer': [final_answer],
            }
            return render_template('index.html', **kwargs)

        print(answer)
        kwargs = {
            'query': question,
            'answer': [answer],
        }
        return render_template('index.html', **kwargs)


if __name__ == '__main__':
    qa = QA()
    content = open("./input_file/model.txt", 'r', encoding='utf-8')
    paragraph = " ".join(content.read().splitlines())
    app.run(host="0.0.0.0", port=8000)
