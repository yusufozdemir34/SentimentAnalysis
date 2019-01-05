import logging
import random

import numpy as np
import pandas as pd

from gensim.models import doc2vec
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
# accuracy_score  bize bir modelin doğru bir şekilde eğitilip öğretilmediğini ve genel olarak nasıl performans gösterdiğini söyleyebilir. Ancak, soruna uygulanmasıyla ilgili ayrıntılı bilgi vermemektedir.
# F1 score, testin doğruluğunun ölçme yönetimidir. Hem kesinliğin hem de hatırlamanın bir değerlendirmesidir ve bir F1 skoru 1 iken mükemmel olarak kabul edilir ve 0 iken toplam bir başarısızlık olur.
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def read_dataset(path):
    dataset = pd.read_csv(path, header=0, delimiter="\t")
    x_train, x_test, y_train, y_test = train_test_split(dataset.review, dataset.sentiment, random_state=0, test_size=0.1)
    x_train = label_sentences(x_train, 'Train')
    x_test = label_sentences(x_test, 'Test')
    all_data = x_train + x_test
    return x_train, x_test, y_train, y_test, all_data


def label_sentences(corpus, label_type):
    """
    Gensim'in Doc2Vec uygulaması, her belgenin / paragrafın kendisiyle ilişkilendirilmiş bir etiketi olmasını gerektirir.
     Bunu LabeledSentence yöntemini kullanarak yapıyoruz. Biçim "TRAIN_i" veya "TEST_i" olacaktır; burada "i"
     gözden geçirmenin dummy endeksi.
    """
    labeled = []
    for i, v in enumerate(corpus):
        label = label_type + '_' + str(i)
        labeled.append(doc2vec.LabeledSentence(v.split(), [label]))
    return labeled


def get_vectors(doc2vec_model, corpus_size, vectors_size, vectors_type):
    """
    Eğitimli doc2vec modelinden vektörler al
     : param doc2vec_model: Eğitimli Doc2Vec modeli
     : param corpus_size: Verinin boyutu
     : param vectors_size: Gömülü vektörlerin boyutu
     : param vectors_type: Eğitim veya Vektörleri Test Etme
     : return: vektörlerin listesi
    """
    vectors = np.zeros((corpus_size, vectors_size))
    for i in range(0, corpus_size):
        prefix = vectors_type + '_' + str(i)
        vectors[i] = doc2vec_model.docvecs[prefix]
    return vectors


def train_doc2vec(corpus):
    logging.info("Doc2Vec sozluk oluşturma")
    d2v = doc2vec.Doc2Vec(min_count=1,  # Toplam frekansı bundan daha düşük olan tüm kelimeleri yok sayar.
                          window=10,  # Bir cümle içinde geçerli ve öngörülen kelime arasındaki maksimum mesafe
                          vector_size=300,  # vektörlerinin boyutsallığı
                          workers=5,  # Modeli eğitmek için çalışan iş parçacığı sayısı
                          alpha=0.025,  # Ilk öğrenme oranı
                          min_alpha=0.00025,  # Eğitim ilerledikçe öğrenme oranı doğrusal olarak min_alpha'ya düşer
                          dm=1)  # dm, eğitim algoritmasını tanımlar. Dm = 1 ise ‘dağıtılmış hafıza’ (PV-DM)
                                 # dm = 0, "dağıtılmış sözcük çuvalı" anlamına gelir (PV-DBOW)
    d2v.build_vocab(corpus)

    logging.info("Doc2Vec model egitimi")
    # 10 epoch yaklasik 15 dakika surebilir. Islemici gucune bagli olarak epoch arttirilabilir
    for epoch in range(10):
        logging.info('Egitim iterasyon #{0}'.format(epoch))
        d2v.train(corpus, total_examples=d2v.corpus_count, epochs=d2v.iter)
        # corpus u karistir
        random.shuffle(corpus)
        # ogrenme oranini dusur
        d2v.alpha -= 0.0002
        # ogrenme oranini sabitle
        d2v.min_alpha = d2v.alpha

    logging.info("Doc2Vec model kaydedildi")
    d2v.save("d2v.model")
    return d2v


def train_classifier(d2v, training_vectors, training_labels):
    logging.info("Siniflandirma egitimi")
    train_vectors = get_vectors(d2v, len(training_vectors), 300, 'Train')
    model = LogisticRegression()
    model.fit(train_vectors, np.array(training_labels))
    training_predictions = model.predict(train_vectors)
    logging.info('Eğitimde Tahmin edilecek sınıflar: {}'.format(np.unique(training_predictions)))
    logging.info('accuracy ile egitim: {}'.format(accuracy_score(training_labels, training_predictions)))
    logging.info('F1 score ile egitim: {}'.format(f1_score(training_labels, training_predictions, average='weighted')))
    return model


def test_classifier(d2v, classifier, testing_vectors, testing_labels):
    logging.info("Siniflandirma test ediliyor")
    test_vectors = get_vectors(d2v, len(testing_vectors), 300, 'Test')
    testing_predictions = classifier.predict(test_vectors)
    logging.info('Tahmin edilen sınıfları test etme: {}'.format(np.unique(testing_predictions)))
    logging.info('Accuracy ile Test: {}'.format(accuracy_score(testing_labels, testing_predictions)))
    logging.info('F1 score ile test: {}'.format(f1_score(testing_labels, testing_predictions, average='weighted')))


if __name__ == "__main__":
    x_train, x_test, y_train, y_test, all_data = read_dataset('yusuf_dataset.csv')
    d2v_model = train_doc2vec(all_data)
    classifier = train_classifier(d2v_model, x_train, y_train)
    test_classifier(d2v_model, classifier, x_test, y_test)
