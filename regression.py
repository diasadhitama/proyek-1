import numpy as np
#from sklearn.externals 
import joblib
from sklearn.linear_model import LinearRegression

X= np.array([40,20,25,20,30,50,40,20,50,40,25,50]).reshape((-1, 1))
Y= np.array([385,400,395,365,475,440,490,420,560,525,480,510])

model = LinearRegression().fit(X,Y)

filename = 'model.sav'
joblib.dump(model, filename)

loaded_model = joblib.load(filename)
Xtest= np.array([40,20,25]).reshape((-1, 1))


loaded_model.predict(Xtest)