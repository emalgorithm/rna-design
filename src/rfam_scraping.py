from ftplib import FTP
import gzip
import os


def get_family_rna_sequences(family_id):
    server = 'ftp.ebi.ac.uk'
    directory = 'pub/databases/Rfam/CURRENT/fasta_files/'
    file_name = '{}.fa.gz'.format(family_id)
    print(file_name)

    # Connect to ftp server
    ftp = FTP(server)
    ftp.login()
    ftp.cwd(directory)

    # Download gz file
    gz_temp_file = '../data/family_rna_sequences/{}.gz'.format(family_id)
    with open(gz_temp_file, 'wb') as f:
        ftp.retrbinary('RETR ' + file_name, f.write)

    # Extract sequences from file
    sequences = extract_sequences(gz_temp_file)

    # Delete file
    os.remove(gz_temp_file)

    return sequences


def store_family_rna_sequences(family_id):
    sequences = get_family_rna_sequences(family_id)
    with open('../data/family_rna_sequences/{}.txt'.format(family_id), 'w') as f:
        for i, seq in enumerate(sequences):
            f.write(seq.decode('ASCII'))
            if i != len(sequences) - 1:
                f.write('\n')


def extract_sequences(fa_file_path):
    sequences = []
    with gzip.open(fa_file_path, 'rb') as f:
        for i, line in enumerate(f):
            if i % 2 != 0:
                sequence = line.strip()
                sequences.append(sequence)

    return sequences
