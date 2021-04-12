import pytcga

studies = pytcga.load_studies()
clinical = pytcga.load_clinical_data('luad')
luad_mutations = \
    pytcga.load_mutation_data(disease_code='LUAD', with_clinical=True)
luad_rnaseq = \
    pytcga.load_rnaseq_data(disease_code='LUAD', with_clinical=True)