from flask import Flask
from flask import request
from transformers import pipeline, AutoTokenizer, RobertaForSequenceClassification, RobertaConfig

app = Flask(__name__)

# Load pipeline
config = RobertaConfig.from_pretrained(
    "/kaggle/input/content-censor/model.bin/config.json",
    from_tf=False,
    num_labels=4,
    output_hidden_states=False,
)
model = RobertaForSequenceClassification.from_pretrained(
    "/kaggle/input/content-censor/model.bin/pytorch_model.bin",
    config=config
)
tokenizer = AutoTokenizer.from_pretrained(
    '/kaggle/input/content-censor/phobert-base/')

# Create the pipeline
classifier = pipeline("text-classification",
                      model=model,
                      tokenizer=tokenizer)


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/predict", methods=['POST'])
def predict(data):
    data = request.form['data']
    outputs = classifier(data)
    return outputs
