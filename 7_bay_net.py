import numpy as np
import pandas as pd
from urllib.request import urlopen
url = 'http://tinyurl.com/easypezy'
names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'heartdisease']
data = pd.read_csv(urlopen(url), names=names)
del data['ca']
del data['slope']
del data['thal']
del data['oldpeak']
data = data.replace('?', np.nan)
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
model = BayesianModel([('age','trestbps'),('age','fbs'),('sex','trestbps'),('sex','trestbps'),('exang','trestbps'),('trestbps','heartdisease'),('fbs','heartdisease'),('heartdisease','restecg'),
('heartdisease','thalach'),('heartdisease','chol')])
model.fit(data, estimator=MaximumLikelihoodEstimator)
print(model.get_cpds('age'))
print(model.get_cpds('sex'))
print(model.get_cpds('chol'))
model.get_independencies()
from pgmpy.inference import VariableElimination
infer = VariableElimination(model)
q = infer.query(variables=['heartdisease'], evidence={'age':28})
print(q['heartdisease'])
q = infer.query(variables=['heartdisease'], evidence={'chol':100})
print(q['heartdisease'])