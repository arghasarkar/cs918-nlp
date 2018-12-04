#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import testsets
import evaluation

# TODO: load training data

for classifier in ['myclassifier1', 'myclassifier2', 'myclassifier3']: # You may rename the names of the classifiers to something more descriptive
    print(classifier)
    if classifier == 'myclassifier1':
        print('Training ' + classifier)
        # TODO: extract features for training classifier1
        # TODO: train sentiment classifier1
    elif classifier == 'myclassifier2':
        print('Training ' + classifier)
        # TODO: extract features for training classifier2
        # TODO: train sentiment classifier2
    elif classifier == 'myclassifier3':
        print('Training ' + classifier)
        # TODO: extract features for training classifier3
        # TODO: train sentiment classifier3

    for testset in testsets.testsets:
        # TODO: classify tweets in test set

        predictions = {'163361196206957578': 'neutral',
                       '768006053969268950': 'negative',
                       '742616104384772304': 'negative',
                       '102313285628711403': 'negative',
                       '653274888624828198': 'negative',
                       '364323072843019872': 'positive',
                       '063115054245201986': 'positive'}
        # TODO: Remove this line, 'predictions' should be populated with the outputs of your classifier

        # predictions = {}
        print(testset)


        evaluation.evaluate(predictions, testset, classifier)

        evaluation.confusion(predictions, testset, classifier)
