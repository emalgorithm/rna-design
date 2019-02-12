from rfam_scraper import RfamScraper

n_families = 3016
family_ids = ['RF{}'.format(str(i).zfill(5)) for i in range(1, n_families + 1)]

rfam_scraper = RfamScraper()
sequences = rfam_scraper.store_families_rna_sequences(family_ids)
