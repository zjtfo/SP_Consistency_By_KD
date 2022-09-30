# KQA Pro version 1.0

KQA Pro is a large-scale dataset of complex question answering over knowledge base. The questions are very diverse and challenging, requiring multiple reasoning capabilities including compositional reasoning, multi-hop reasoning, quantitative comparison, set operations, and etc. Strong supervisions of SPARQL and program are provided for each question.


## Usage
There are four json files included in dataset:

- `kb.json`, the target knowledge base used to answer questions, which is a dense subset of [Wikidata](https://www.wikidata.org/wiki/Wikidata:Main_Page).
- `train.json`, the training set, including 94,376 QA pairs with annotations of SPARQL and program for each.
- `val.json`, the validation set, including 11,797 QA pairs with SPARQL and program.
- `test.json`, the test set, including 11,797 questions, with 10 candidate choices for each. 

Following is the detailed formats

**kb.json**
```
{
    'concepts':
    {
        '<id>':
        {
            'name': str,
            'instanceOf': ['<id>', '<id>'], # ids of parent concept
        }
    },
    'entities': # excluding concepts
    {
        '<id>': 
        {
            'name': str,
            'instanceOf': ['<id>', '<id>'], # ids of parent concept
            'attributes':
            [
                {
                    'key': str, # attribute key
                    'value':  # attribute value
                    {
                        'type': 'string'/'quantity'/'date'/'year',
                        'value': float/int/str, # float or int for quantity, int for year, 'yyyy/mm/dd' for date
                        'unit': str,  # for quantity
                    },
                    'qualifiers':
                    {
                        '<qk>':  # qualifier key, one key may have multiple corresponding qualifier values
                        [
                            {
                                'type': 'string'/'quantity'/'date'/'year',
                                'value': float/int/str,
                                'unit': str,
                            }, # the format of qualifier value is similar to attribute value
                        ]
                    }
                },
            ]
            'relations':
            [
                {
                    'predicate': str,
                    'object': '<id>', # NOTE: it may be a concept id
                    'direction': 'forward'/'backward',
                    'qualifiers':
                    {
                        '<qk>':  # qualifier key, one key may have multiple corresponding qualifier values
                        [
                            {
                                'type': 'string'/'quantity'/'date'/'year',
                                'value': float/int/str,
                                'unit': str,
                            }, # the format of qualifier value is similar to attribute value
                        ]
                    }
                },
            ]
        }
    }
}
```

**train.json/val.json**
```
[
    {
        'question': str,
        'sparql': str, # executable in our virtuoso engine
        'program': 
        [
            {
                'function': str,  # function name
                'dependencies': [int],  # functional inputs, representing indices of the preceding functions
                'inputs': [str],  # textual inputs
            }
        ],
        'choices': [str],  # 10 answer choices
        'answer': str,  # golden answer
    }
]
```

**test.json**
```
[
    {
        'question': str,
        'choices': [str],  # 10 answer choices
    }
]
```
