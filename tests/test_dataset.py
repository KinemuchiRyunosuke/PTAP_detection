import numpy as np
import unittest
import json
from Bio import SeqIO

from src.dataset import Dataset, classify_records


class TestDataset(unittest.TestCase):
    def setUp(self):
        motif_data_path = 'tests/test_motif_data.json'

        length = 10

        with open(motif_data_path, 'r') as f:
            motif_data = json.load(f)
        motif_data = motif_data[0]

        self.dataset = Dataset(motif_data['motifs'],
                               protein_subnames=motif_data['protein_subnames'],
                               length=length)

    def test_n_gram_split(self):
        seq = 'NNPQQQQRNNFGHIJKNNNNNNFGHI'
        label_list = [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, \
                      3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3]

        x, y = self.dataset._n_gram_split(seq, label_list)

        correct_x = ['NNPQQQQRNN', 'NPQQQQRNNF', 'PQQQQRNNFG', 'QQQQRNNFGH',
                     'QQQRNNFGHI', 'QQRNNFGHIJ', 'QRNNFGHIJK', 'RNNFGHIJKN',
                     'NNFGHIJKNN', 'NFGHIJKNNN', 'FGHIJKNNNN', 'GHIJKNNNNN',
                     'HIJKNNNNNN', 'IJKNNNNNNF', 'JKNNNNNNFG', 'KNNNNNNFGH',
                     'NNNNNNFGHI']
        correct_y = [1, 1, 1, 1, 0, 1, 1, 1, \
                     1, 1, 1, 1, 1, 0, 0, 0, 0]
        self.assertEqual(x, correct_x)
        self.assertEqual(y, correct_y)


class TestDataset2(unittest.TestCase):
    def setUp(self):
        length = 10
        separate_len = 1
        fasta_path= 'tests/test_fasta.fasta'

        motif_data_path = "tests/test_motif_data.json"
        with open(motif_data_path, 'r') as f:
            motif_data = json.load(f)
            motif_data = motif_data[0]

        with open(fasta_path, 'r') as f:
            self.records = [record for record in SeqIO.parse(f, 'fasta')]

        self.dataset = Dataset(
                motifs=motif_data['motifs'],
                protein_subnames=motif_data['protein_subnames'],
                length=length,
                separate_len=separate_len)

    def test_annotate(self):
        label_lists = list(map(self.dataset._annotate, self.records))
        correct_label_lists = \
            [[0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        self.assertEqual(label_lists, correct_label_lists)

    def test_make_dataset(self):
        correct_dataset = {
            'env': (
                ['N N P Q Q Q A R N N', 'N P Q Q Q A R N N A', 'P Q Q Q A R N N A B',
                 'Q Q Q A R N N A B C', 'Q Q A R N N A B C C', 'Q A R N N A B C C C',
                 'A R N N A B C C C N', 'R N N A B C C C N N', 'N N A B C C C N N F',
                 'N A B C C C N N F G', 'A B C C C N N F G H', 'B C C C N N F G H I',
                 'C C C N N F G H I J', 'C C N N F G H I J K'],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]),
            'tat': (
                ['N N P Q Q Q A R N N', 'N P Q Q Q A R N N A', 'P Q Q Q A R N N A B',
                 'Q Q Q A R N N A B C', 'Q Q A R N N A B C C', 'Q A R N N A B C C C',
                 'A R N N A B C C C N', 'R N N A B C C C N N', 'N N A B C C C N N F',
                 'N A B C C C N N F G', 'A B C C C N N F G H', 'B C C C N N F G H I',
                 'C C C N N F G H I J', 'C C N N F G H I J K'],
                [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            'rev': (
                ['N N P Q Q Q A R N N', 'N P Q Q Q A R N N A', 'P Q Q Q A R N N A B',
                 'Q Q Q A R N N A B C', 'Q Q A R N N A B C C', 'Q A R N N A B C C C',
                 'A R N N A B C C C N', 'R N N A B C C C N N', 'N N A B C C C N N F',
                 'N A B C C C N N F G', 'A B C C C N N F G H', 'B C C C N N F G H I',
                 'C C C N N F G H I J', 'C C N N F G H I J K'],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        }

        dataset = self.dataset.make_dataset(self.records)
        self.assertTrue(dataset == correct_dataset)


class TestDataset3(unittest.TestCase):
    def setUp(self):
        length = 10
        separate_len = 2
        fasta_path= 'tests/test_fasta.fasta'

        motif_data_path = "tests/test_motif_data.json"
        with open(motif_data_path, 'r') as f:
            motif_data = json.load(f)
            motif_data = motif_data[0]

        with open(fasta_path, 'r') as f:
            self.records = [record for record in SeqIO.parse(f, 'fasta')]

        self.dataset = Dataset(
                motifs=motif_data['motifs'],
                protein_subnames=motif_data['protein_subnames'],
                length=length,
                separate_len=separate_len)

    def test_annotate(self):
        label_lists = list(map(self.dataset._annotate, self.records))
        correct_label_lists = \
            [[0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        self.assertEqual(label_lists, correct_label_lists)

    def test_make_dataset(self):
        correct_dataset = {
            'env': (
                ['NN NP PQ QQ QQ QA AR RN NN', 'NP PQ QQ QQ QA AR RN NN NA',
                 'PQ QQ QQ QA AR RN NN NA AB', 'QQ QQ QA AR RN NN NA AB BC',
                 'QQ QA AR RN NN NA AB BC CC', 'QA AR RN NN NA AB BC CC CC',
                 'AR RN NN NA AB BC CC CC CN', 'RN NN NA AB BC CC CC CN NN',
                 'NN NA AB BC CC CC CN NN NF', 'NA AB BC CC CC CN NN NF FG',
                 'AB BC CC CC CN NN NF FG GH', 'BC CC CC CN NN NF FG GH HI',
                 'CC CC CN NN NF FG GH HI IJ', 'CC CN NN NF FG GH HI IJ JK'],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]),
            'tat': (
                ['NN NP PQ QQ QQ QA AR RN NN', 'NP PQ QQ QQ QA AR RN NN NA',
                 'PQ QQ QQ QA AR RN NN NA AB', 'QQ QQ QA AR RN NN NA AB BC',
                 'QQ QA AR RN NN NA AB BC CC', 'QA AR RN NN NA AB BC CC CC',
                 'AR RN NN NA AB BC CC CC CN', 'RN NN NA AB BC CC CC CN NN',
                 'NN NA AB BC CC CC CN NN NF', 'NA AB BC CC CC CN NN NF FG',
                 'AB BC CC CC CN NN NF FG GH', 'BC CC CC CN NN NF FG GH HI',
                 'CC CC CN NN NF FG GH HI IJ', 'CC CN NN NF FG GH HI IJ JK'],
                [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            'rev': (
                ['NN NP PQ QQ QQ QA AR RN NN', 'NP PQ QQ QQ QA AR RN NN NA',
                 'PQ QQ QQ QA AR RN NN NA AB', 'QQ QQ QA AR RN NN NA AB BC',
                 'QQ QA AR RN NN NA AB BC CC', 'QA AR RN NN NA AB BC CC CC',
                 'AR RN NN NA AB BC CC CC CN', 'RN NN NA AB BC CC CC CN NN',
                 'NN NA AB BC CC CC CN NN NF', 'NA AB BC CC CC CN NN NF FG',
                 'AB BC CC CC CN NN NF FG GH', 'BC CC CC CN NN NF FG GH HI',
                 'CC CC CN NN NF FG GH HI IJ', 'CC CN NN NF FG GH HI IJ JK'],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        }

        dataset = self.dataset.make_dataset(self.records)
        self.assertTrue(dataset == correct_dataset)


class Testdataset4(unittest.TestCase):
    def setUp(self):
        self.length = 4

        separate_len = 1
        fasta_path= 'tests/test_fasta.fasta'
        rm_positive_neighbor = 2

        motif_data_path = "tests/test_motif_data.json"
        with open(motif_data_path, 'r') as f:
            motif_data = json.load(f)
            motif_data = motif_data[0]

        with open(fasta_path, 'r') as f:
            self.records = [record for record in SeqIO.parse(f, 'fasta')]

        self.dataset = Dataset(
                motifs=motif_data['motifs'],
                protein_subnames=motif_data['protein_subnames'],
                length=self.length,
                separate_len=separate_len,
                rm_positive_neighbor=rm_positive_neighbor)

    def test_rm_positive_neighbor(self):
        x = ['ABCD', 'BCDE', 'CDEF', 'DEFG', 'EFGH', 'FGHI', 'GHIJ', \
             'HIJK', 'IJKL', 'JKLM', 'KLMN', 'LMNO', 'MNOP', 'NOPQ', \
             'OPQR', 'PQRS', 'QRST', 'RSTU', 'STUV', 'TUVW', 'UVWX']
        y = [0, 0, 0, 1, 1, 1, 1, \
             0, 0, 0, 0, 0, 1, 1, \
             1, 0, 0, 0, 0, 0, 1]

        correct_x = \
            ['ABCD', 'DEFG', 'EFGH', 'FGHI', 'GHIJ', \
             'JKLM', 'MNOP', 'NOPQ', \
             'OPQR', 'RSTU', 'UVWX']
        correct_y = [0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1]

        x, y = self.dataset._rm_positive_neighbor(x, y)
        self.assertEqual(x, correct_x)
        self.assertEqual(y, correct_y)


class TestClassifyRecords(unittest.TestCase):
    def setUp(self):
        motif_data_path = "tests/test_motif_data.json"
        with open(motif_data_path, 'r') as f:
            motif_data = json.load(f)
        self.protein_subnames = motif_data[0]['protein_subnames']

        fasta_path= 'tests/test_fasta.fasta'
        with open(fasta_path, 'r') as f:
            self.records = [record for record in SeqIO.parse(f, 'fasta')]

    def test_classify_records(self):
        records_dict = classify_records(self.records, self.protein_subnames)

        keys = ['tat', 'env', 'rev']
        for key, records in records_dict.items():
            if key in keys:
                self.assertEqual(len(records), 1)
            else:
                self.assertEqual(len(records), 0)


