from scripts.data import getdata
from scripts.model import create_cluster, benchmark, save_model

diabetes, stroke, hypertension = getdata('data')
print(diabetes.columns)

# Unsupervised learning to find metabolism profiles
diabetes = create_cluster(diabetes)

# Supervised learning to have a production point attributor
model = benchmark(diabetes)

save_model(model, filename='meta_machina.pkl')


