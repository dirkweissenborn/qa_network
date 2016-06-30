"""
This file contains abstractions for defining queries and its corresponding support in form of
supporting query-answer pairs. Each context (list of symbol ids) can contain multiple
cloze-style queries.

A query can have additional support which is a list of additional query-answer pairs in the same
format as normal queries. They can be used to help answer the original query.


"""

class ContextQueries:
    """
    Wraps all queries for a certain context into a single object. If used as support, context can be specified
    as None which means that the context of the actual query serves as context for the support.
    """

    def __init__(self, context, queries, support=None, collaborative_support=False, source=None):
        """
        :param context: list of symbol ids that should be the same for all queries
        :param queries: ClozeQueries within this context.
        :param support: None or list of ContextQueries; Support defined here serves as support for all
        queries of this object.
        :param collaborative_support: Used for answering queries collaboratively (has never been used yet)
        :param source: optional, can be used to keep track of the context origin
        """
        assert all((isinstance(q, ClozeQuery) and q.context == context for q in queries)), \
            "Context queries must share same context."
        self.context = context
        self.queries = queries
        self.collaborative_support = collaborative_support
        #supporting evidence for all queries
        assert support is None or all((isinstance(q, ContextQueries) for q in support)), \
            "Support must be a list of ContextQueries"
        self.support = support
        self.source = source  # information about source of this query (optional)


class ClozeQuery:
    """
    Cloze-queries are fully defined by a a span-of-interest (start and end) within a certain context.
    Note, the qa model embeds a query by its surrounding context (everything outside the span). If you
    want the model to also encode the span itself, simply use the negative span, i.e. swap start and end.
    If answer is None, only candidates will be scored in the QANetwork. If answer is defined the model will
    score "[answer] + candidates".
    """

    def __init__(self, context, start, end, answer, answer_word, candidates, support=None):
        """
        :param context: list of symbol ids
        :param start: start of the span of the query
        :param end: end of the span of the query
        :param answer: id of the answer to be predicted for the given span; If None only candidates are scored
        in QA Network
        :param answer_word: symbol-id for the respective answer that are used to refine query between hops
        (note, answer vocab can differ from input vocab, thus the differentiation between answer and answer_word)
        :param candidates: answer candidates of this query without the answer itself
        :param support: None or list of ContextQueries
        """
        self.context = context
        self.start = start
        self.end = end
        self.answer = answer
        self.answer_word = answer_word
        self.candidates = candidates
        assert support is None or all((isinstance(q, ContextQueries) for q in support)), \
            "Support must be a list of ContextQueries"
        self.support = support


def flatten_queries(context_queries_list):
    ret = []
    for qs in context_queries_list:
        ret.extend(qs.queries)
    return ret
