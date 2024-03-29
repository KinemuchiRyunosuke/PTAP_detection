import numpy as np


class Dataset:
    def __init__(self, motifs, protein_subnames, length=10,
                 remove_X=True, separate_len=None, rm_positive_neighbor=0,
                 motif_neighbor=0):
        self.motifs = motifs        # dict: アノテーションするmotif配列情報

        # dict: タンパク質の種別とその異名をまとめた辞書
        self.protein_subnames = protein_subnames

        self.length = length        # int: 断片の長さ

        self.remove_X = remove_X    # bool: 未知アミノ酸Xが含まれる
                                    #       配列を無視する．

        # int: n連続アミノ酸でベクトルを生成するときは長さを指定する．
        #   Noneを指定するとn連続アミノ酸頻度に分割しない．
        self.separate_len = separate_len

        # int: 陽性となった断片の近傍n個のデータセットを除去する．
        self.rm_positive_neighbor = rm_positive_neighbor

        # int: モチーフ周辺n残基を含めた配列を完全に含む配列断片を陽性とする
        self.motif_neighbor = motif_neighbor

    def make_dataset(self, records, test_mode=False):
        """ タンパク質毎にアミノ酸断片とラベルを生成

        Arg:
            records: FASTAファイルのデータ

        Return:
            dict(protein_name: (x, y))    (x, y: ndarray)
                タンパク質の種別ごとに整理されたアミノ酸断片xとそのラベルy

        """
        protein_list = list(self.protein_subnames.keys())
        dataset = {protein: None for protein in protein_list}

        for protein in protein_list:
            xs, ys = [], []

            # テストモードの場合，計算量短縮のためにループ回数を制限
            if test_mode:
                test_count = 0

            for record in records:
                if self._get_protein_name(record) != protein:
                    continue

                if self.remove_X:
                    if 'X' in record.seq:
                        continue

                if len(record.seq) < self.length:
                    continue

                if test_mode:
                    test_count += 1
                    if test_count > 100:
                        break

                label_list = self._annotate(record)

                x, y = self._n_gram_split(record.seq, label_list)

                if self.separate_len is not None:
                    x = self._separate(x, n=self.separate_len)

                xs += x
                ys += y

            dataset[protein] = (xs, ys)

        return dataset

    def _n_gram_split(self, seq, label_list):
        """ アミノ酸配列を分割し，n_gramのデータセットを作成

        Args:
            seq(str): アミノ酸配列
            label_list(list of int): モチーフでない箇所を0,
                モチーフである箇所をそのモチーフのidで示したリスト

        Returns:
            x(list of str): アミノ酸断片
            y(list of int): アミノ酸断片にmotifが完全に含まれる場合は0,
                含まれない場合は1

        """
        x, y = [], []
        i = 0

        while (i <= len(seq) - self.length):
            fragment = seq[i:(i + self.length)]
            x.append(fragment)
            y.append(int(self._has_motif(label_list[i:(i + self.length)])))
            i += 1

        if self.rm_positive_neighbor > 0:
            x, y = self._rm_positive_neighbor(x, y)

        return x, y

    def _has_motif(self, labels):
        """ アミノ酸断片がモチーフを保有する場合はTrueを返す """
        has_motif = False

        # SLiM周辺のself.motif_neighbor残基を陽性としてみなす
        expanded_labels = labels.copy()
        for i in range(len(labels)):
            if labels[i] != 0:
                for j in range(self.motif_neighbor):
                    expanded_labels[max(i - (j + 1), 0)] = labels[i]
                    expanded_labels[min(i + (j + 1), len(labels) - 1)] = \
                            labels[i]

        ids = list(set(expanded_labels))
        ids = [id for id in ids if id != 0]

        if ids != []:
            for id in ids:
                threshold = len(self.motifs[id - 1]['motif_seq']) + \
                    2 * self.motif_neighbor

                count = 0
                for label in expanded_labels:
                    if label == id:
                        count += 1
                    else:
                        count = 0

                    if count >= threshold:
                        has_motif = True
                        break

        return has_motif

    def _rm_positive_neighbor(self, x, y):
        rm_labels = [False] * len(y)
        for i, label in enumerate(y):
            if label == 0:
                continue

            for j in range(self.rm_positive_neighbor):
                rm_id_up = max(0, i - j - 1)
                rm_id_down = min(len(y) - 1, i + j + 1)

                if y[rm_id_up] == 0:
                    rm_labels[rm_id_up] = True

                if y[rm_id_down] == 0:
                    rm_labels[rm_id_down] = True

        x = [seq for i, seq in enumerate(x) if not rm_labels[i]]
        y = [label for i, label in enumerate(y) if not rm_labels[i]]

        return x, y

    def _annotate(self, record, ignore_not_motif_protein=True):
        """ ラベルリストを作成

        次のようなアミノ酸配列から'APTAPP'というmotif配列を検出する場合，
            sequence : MGAPTAPPQDN
            label    : 00111111000
        となるから，ラベルリストは次のようになる．
            label_list = [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]

        Arg:
            record: アミノ酸配列情報とヘッダー行を格納したFASTAファイル情報
            ignore_not_motif_protein(bool):
                Trueの場合，モチーフを保有することがわかっているタンパク質に
                対してのみアノテーションを行う．

        Return:
            label_list(list of int): モチーフに該当する部分をモチーフの番号,
                該当しない部分を0としたリスト．

        """
        # self.motifsに対して最大・最小モチーフ長を求める
        max_subsequence_len, min_subsequence_len = 0, 99999
        for motif_data in self.motifs:
            subsequence = motif_data['motif_upstream_seq'] + \
                          motif_data['motif_seq'] + \
                          motif_data['motif_downstream_seq']
            max_subsequence_len = max(len(subsequence), max_subsequence_len)
            min_subsequence_len = min(len(subsequence), min_subsequence_len)

        if ignore_not_motif_protein:
            motif_ids = self._is_motif_protein(record)
        else:
            motif_ids = [i + 1 for i in range(len(self.motifs))]

        if motif_ids == []:
            label_list = [0] * len(record.seq)
        else:
            label_list = []
            i = 0

            while(i < len(record.seq) - min_subsequence_len + 1):
                fragment = record.seq[i:(i + max_subsequence_len)]
                label_id = self._annotate_one(fragment, motif_ids)

                if label_id in motif_ids:
                    motif = self.motifs[label_id - 1]
                    label_list += [0] * len(motif['motif_upstream_seq'])
                    label_list += [label_id] * len(motif['motif_seq'])
                    label_list += [0] * len(motif['motif_downstream_seq'])

                    subsequence_len = len(motif['motif_upstream_seq']) + \
                                      len(motif['motif_seq']) + \
                                      len(motif['motif_downstream_seq'])
                    i += subsequence_len

                else:
                    label_list.append(0)
                    i += 1

            while (i < len(record.seq)):
                label_list.append(0)
                i += 1

        return label_list

    def _annotate_one(self, fragment, motif_ids):
        """ アミノ酸断片がモチーフ配列である場合モチーフの番号,
            そうでない場合は0を返す

        Args:
            fragment(str): アミノ酸断片．
            motif_ids: 指定された番号のモチーフに対してだけアノテーションを行う

        Return:
            int: アミノ酸断片がmotif_dataに記されたモチーフと一致した場合は
                モチーフの番号（motif_data.jsonに登録されている順番）を，
                一致しない場合は0を返す．

        """
        label_id = 0
        for motif_id in motif_ids:
            motif_data = self.motifs[motif_id - 1]

            subsequence = motif_data['motif_upstream_seq'] + \
                          motif_data['motif_seq'] + \
                          motif_data['motif_downstream_seq']
            subsequence_length = len(subsequence)

            if len(fragment) < subsequence_length:
                continue

            threshold = subsequence_length - motif_data['replacement_tolerance']

            count = 0
            for i, subsequence_char in enumerate(subsequence):
                if subsequence_char == fragment[i]:
                    count += 1

            if count >= threshold:
                label_id = motif_id
                break

        return label_id

    def _is_motif_protein(self, record):
        """ レコードのヘッダー行を参照し，モチーフを保有するタンパク質である場合は
            モチーフの番号をリストにして返す

        Arg:
            desc(str): FASTAファイルのレコード

        Return:
            list of int: モチーフを保有するタンパク質である場合，そのタンパク質で
                登録されているモチーフの番号をリストにして返す

        """
        motif_ids = []
        for motif_id, motif_data in enumerate(self.motifs):
            if motif_data['protein'] == self._get_protein_name(record):
                motif_ids.append(motif_id + 1)

        return motif_ids

    def _separate(self, seqs, n=2):
        """ n残基ずつの断片に分けて，' 'で区切る

        Args:
            seqs(ndarray, list): 操作を行う配列
            n(int): n残基ずつに分ける

        Returns:
            list of str

        """
        if type(seqs).__module__ == 'numpy':
            seqs = np.squeeze(seqs)
            seqs = seqs.tolist()

        separated_seqs = []
        for seq in seqs:
            fragments = []
            for i in range(len(seq) - n + 1):
                fragments.append(str(seq[i:i+n]))

            separated_seqs.append(' '.join(fragments))

        return separated_seqs

    def _get_protein_name(self, record):
        """ recordのタンパク質種を特定する

        Args:
            record(list): FASTAファイルのデータ．

        Return:
            str: 特定したタンパク質名.
                特定できない場合はNoneを返す

        """
        desc = record.description
        desc = desc[(len(record.id) + 1):]  # accession_idを削除
        desc = desc.lower()

        for protein_name, subnames in self.protein_subnames.items():
            for subname in subnames:
                if subname.lower() in desc:
                    return protein_name

        return None


def classify_records(records, protein_subnames):
    """ recordをタンパク質ごとに分類する
    Args:
        records(list): recordのリスト
        protein_subnames(dict): タンパク質の分類と，その異名を格納した辞書

    Return:
        dict: (タンパク質名):(recordのリスト)

    """
    classified_records = {key:[] for key in protein_subnames.keys()}
    classified_records['others'] = []

    for record in records:
        desc = record.description
        desc = desc[(len(record.id) + 1):]  # accession_idを削除
        desc = desc.lower()

        for protein_name, subnames in protein_subnames.items():
            finish = False

            for subname in subnames:
                if subname.lower() in desc:
                    classified_records[protein_name].append(record)
                    finish = True
                    break

            if finish:
                break

        if not finish:
            classified_records['others'].append(record)

    return classified_records
