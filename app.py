from flask import Flask, render_template,request
from transformers import pipeline, set_seed
from summarizer import Summarizer
import legal_text_summarization
import tokenizer
from transformers import pipeline
from transformers import AutoTokenizer

app = Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained("tokenizer/")


@app.route("/")
def msg():
    return render_template('index.html')

@app.route("/summarize",methods=['POST','GET'])
def getSummary():
    reference = request.form['data']
    length = int(request.form['len'])
    gen_kwargs = {"length_penalty": 0.8, "num_beams":8, "max_length": length}
    pipe1 = pipeline("summarization", model="legal_text_summarization",tokenizer=tokenizer)
    result = pipe1(reference, **gen_kwargs)[0]["summary_text"]
    return render_template('summary.html',reference=reference, result=result)

if __name__ =="__main__":
    app.run(debug=True,port=8000)