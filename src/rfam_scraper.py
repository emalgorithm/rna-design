from ftplib import FTP
import gzip
import os
import pickle


class RfamScraper:
    def __init__(self):
        self.server = 'ftp.ebi.ac.uk'
        self.directory = 'pub/databases/Rfam/CURRENT/fasta_files/'

        # Connect to ftp server
        self.ftp = FTP(self.server)
        self.ftp.login()
        self.ftp.cwd(self.directory)

    def get_family_rna_sequences(self, family_id):
        file_name = '{}.fa.gz'.format(family_id)
        # Download gz file
        gz_temp_file = '../data/family_rna_sequences/{}.gz'.format(family_id)
        with open(gz_temp_file, 'wb') as f:
            try:
                self.ftp.retrbinary('RETR ' + file_name, f.write)
            except:
                print("Failed to download file for family {}".format(family_id))

        # Extract sequences from file
        sequences = self.extract_sequences(gz_temp_file)

        # Delete file
        os.remove(gz_temp_file)

        return sequences

    def store_families_rna_sequences(self, family_ids):
        for family_id in family_ids:
            sequences = self.get_family_rna_sequences(family_id)
            # with open('../data/family_rna_sequences/{}.txt'.format(family_id), 'w') as f:
            #     for i, seq in enumerate(sequences):
            #         f.write(seq)
            #         if i != len(sequences) - 1:
            #             f.write('\n')
            pickle.dump(sequences, open('../data/family_rna_sequences/{}.pkl'.format(family_id),
                                        'wb'))

    def extract_sequences(self, fa_file_path):
        sequences = []
        with gzip.open(fa_file_path, 'rb') as f:
            sequence = ''
            for line in f:
                line = line.strip().decode('ASCII')
                if self.is_line_sequence(line):
                    sequence += line
                else:
                    if sequence != '':
                        sequences.append(sequence)
                        sequence = ''

        return sequences

    @staticmethod
    def is_line_sequence(self, line):
        allowed = ['C', 'A', 'G', 'T']
        for c in line:
            if c not in allowed:
                return False
        return True
