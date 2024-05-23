from scripts.data import getdata,interpret
from scripts.model import create_cluster, benchmark_classification, benchmark_regression, save_model

diabetes, stroke, hypertension = getdata('data')
print(diabetes.columns)

# Unsupervised learning to find metabolism profiles
diabetes = create_cluster(diabetes)

# Supervised learning to have a production point attributor
meta_machina = benchmark_classification(diabetes)

# Collect the data to have production projector models
diabetes, umap_x, umap_y = interpret(diabetes)

# Generate projectors models 
x_projector_machina = benchmark_regression(diabetes,umap_x)
y_projector_machina = benchmark_regression(diabetes,umap_y)

# Save models 
save_model(meta_machina, filename='meta_machina.pkl')

# Save Projectors Models 
save_model(x_projector_machina, filename='x_projector.pkl')
save_model(y_projector_machina, filename='y_projector.pkl')