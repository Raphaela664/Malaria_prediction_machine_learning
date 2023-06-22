from django.shortcuts import render
import joblib
import os
from joblib import load

# Create your views here.
def index(request):
    return render(request,'index.html')

def about(request):
    return render(request,'about.html')
def result(request):
    temperature= float(request.POST.get("temperature"))
    parasite_density= float(request.POST.get("parasite_density"))
    wbc_count = float(request.POST.get("wbc_count"))
    hb_level = float(request.POST.get("hb_level"))
    hematocrit = float(request.POST.get("hematocrit"))
    mean_cell_volume = float(request.POST.get("mean_cell_volume"))
    mean_corp_hb = float(request.POST.get("mean_corp_hb"))
    mean_cell_hb_conc = float(request.POST.get("mean_cell_hb_conc"))
    platelet_count = float(request.POST.get("platelet_count"))
    platelet_distr_width = float(request.POST.get("platelet_distr_width"))
    mean_platelet_vl = float(request.POST.get("mean_platelet_vl"))
    neutrophils_percent = float(request.POST.get("neutrophils_percent"))
    lymphocytes_percent= float(request.POST.get("lymphocytes_percent"))
    mixed_cells_percent= float(request.POST.get("mixed_cells_percent"))
    neutrophils_count= float(request.POST.get("neutrophils_count"))
    lymphocytes_count= float(request.POST.get("lymphocytes_count"))
    mixed_cells_count= float(request.POST.get("mixed_cells_count"))
    
    data_input = [temperature,parasite_density,wbc_count,hb_level,hematocrit,mean_cell_volume,
                  mean_corp_hb,mean_cell_hb_conc,platelet_count,platelet_distr_width, mean_platelet_vl
                  ,neutrophils_percent,lymphocytes_percent,mixed_cells_percent,neutrophils_count,lymphocytes_count,
                  mixed_cells_count]
    #model = joblib.load('C:\\Users\\admin\\Downloads\\EXAM_PREP\\myapp\\recommendation.joblib')
    model_path = os.path.join(os.path.dirname(__file__), 'predictions.joblib')
    model = load(model_path)
    prediction = model.predict([data_input])
    context = {
        'temperature':temperature,
        'parasite_density':parasite_density,
        'wbc_count':wbc_count,
        'hb_level':hb_level,
        'hematocrit':hematocrit,
        'mean_cell_volume':mean_cell_volume,
        'mean_corp_hb':mean_corp_hb,
        'mean_cell_hb_conc':mean_cell_hb_conc,
        'platelet_count':platelet_count,
        'platelet_distr_width':platelet_distr_width,
        'mean_platelet_vl':mean_platelet_vl,
        'neutrophils_percent':neutrophils_percent,
        'lymphocytes_percent':lymphocytes_percent,
        'mixed_cells_percent':mixed_cells_percent,
        'neutrophils_count':neutrophils_count,
        'lymphocytes_count':lymphocytes_count,
        'mixed_cells_count':mixed_cells_count,
        'prediction': prediction
        
        
        
        
    }
    return render(request,'index.html',context)











