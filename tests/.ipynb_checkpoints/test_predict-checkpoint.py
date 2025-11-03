import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from src.ai_crisis.predict import predict_text
from src.ai_crisis.preprocessing import simple_clean

def make_dummy_model():
    X = [
        "flood water rising need help",    # informative (label 1)
        "lost my house in flood",          # informative (label 1)
        "watching movie tonight",          # not informative (label 0)
        "enjoying a nice day at park"      # not informative (label 0)
    ]
    y = [1, 1, 0, 0]
    vect = TfidfVectorizer(max_features=100)
    Xv = vect.fit_transform(X)
    clf = LogisticRegression(random_state=42, max_iter=1000)
    clf.fit(Xv, y)
    return clf, vect

def test_predict_text_returns_label_and_confidence():
    model, vect = make_dummy_model()
    text = "please send rescue, severe flood"
    pred, conf = predict_text(model, vect, text, simple_clean)

    assert pred in model.classes_ or str(pred).isdigit()
    assert isinstance(conf, float)
    assert 0.0 < conf < 1.0

def test_predict_text_confidence_higher_for_informative():
    model, vect = make_dummy_model()
    informative = "we need rescue boats, flood destroyed homes"
    not_inform = "i love this travel destination"
    _, conf_inf = predict_text(model, vect, informative, simple_clean)
    _, conf_not = predict_text(model, vect, not_inform, simple_clean)
    assert conf_inf >= conf_not
