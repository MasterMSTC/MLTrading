# -*- coding: utf-8 -*-

import numpy as np
from itertools import product
from collections import Counter


def get_probabilities(dataframe, list_of_variables):
    """
    It counts number of different events to obtain the probability of each

    Parameters
    ----------
    dataframe : Pandas DataFrame
               Raw data to be processed
    list_of_variables : list
                        labels of the variables

    Returns
    -------
    out : ProbabilityDistribution object
    """

    # Get the different states of each variable
    values_of_variables = {}
    for v in list_of_variables:
        values_of_variables[v] = set(dataframe[v])
    # Combine them to get the states of the variables as a whole
    values_of_x = list(product(*[values_of_variables[v] for v in list_of_variables]))

    # Count how many times each state happens
    results = [tuple(e) for e in dataframe[list_of_variables].values]
    values = Counter(results)

    # Get probabilities for all states
    p = {}
    total = np.sum(list(values.values()))
    for label in values_of_x:
        p[label] = values[label] / total

    return ProbabilityDistribution(p, list_of_variables)


class ConditionalProbabilityDistribution:

    def __init__(self, probabilities, targets, conditionals):
        """
        An object that contains the conditional probabilities based on a labeled probabilities.
         That is, given prbabilities(targets, conditionals) this class calculates the function
         p(targets | conditionals).

        Parameters
        ----------
        probabilities : dictionary
                        Values of the probabilities for the various states. The keys must be tuples
                         as long as the number of inputtted variables. The first variables must be
                         the targets and the last the conditioning variables.
        targets : list
                  Names of the target variables
        conditionals : list
                       Names of the conditional variables
        """

        self.targets = targets
        self.conditionals = conditionals
        variables = targets + conditionals
        self.variables = variables
        self.joint_distribution = ProbabilityDistribution(probabilities, variables)
        self.marginal_distribution = self.joint_distribution.marginalize(targets)
        self.preprocessed_probabilities = False

    def __getitem__(self, events):
        """
        Provides the value of the conditional probability fot a given event

        Parameters
        ----------
        events : tuple of 2 tuples
                 Contains, in the first position, a tuple with the values of the
                  target/non-conditional event and, on the second place, the values of the
                  conditional variables

        Returns
        -------
        out : float
              conditional probability
        """
        if self.preprocessed_probabilities:
            return self.conditional_probabilities[events]
        else:
            targets, conditionals = events
            joint_event = targets + conditionals
            marginal_probability = self.marginal_distribution[conditionals]
            if np.isclose(marginal_probability):
                return 0.
            else:
                return self.joint_distribution[joint_event] / marginal_probability


class ProbabilityDistribution:
    def __init__(self, probabilities, variables=None):
        """
        An object whose value property is returned following the given probability distribution

        Parameters
        ----------
        probabilities : dictionary
                        Values of the probabilities for the various states. If <variables> is
                         provided, i.e. it is not None, the keys must be tuples as long as the
                         number of inputtted variables.
        variables : list, optional
                    Names of the various variables
        """
        # Check that the sum of the probabilities is one.
        if abs(sum(probabilities.values()) - 1.0) > 0.01:
            raise ValueError("Probability must add to 1.0")

        # If variables are labeled or not must be known
        self.are_vars_labeled = variables is not None
        # Check that inputted values are correctly formatted
        if self.are_vars_labeled:
            number_of_vars = None
            for key in probabilities.keys():
                # The number of values must be the same for all states
                if number_of_vars is None:
                    number_of_vars = len(key)
                else:
                    if len(key) != number_of_vars:
                        raise ValueError("All states must be defined by the same number of"
                                         " variables")
                # The number of labels must be equal to the number of inputted variables
                if len(key) != len(variables):
                    raise ValueError("The number of variables must be the same for all of them.")

        # Store values for further use
        self.variables = variables
        self.probabilities = probabilities
        self.keys, self.values = zip(*sorted(list(probabilities.items())))

    def __repr__(self):
        """
        This'll give you something like '70.0% No, 30.0% Yes'
        """
        # return ', '.join(['{:.1f}% {}'.format(100 * self.p[i], i) for i in self.p.keys()])
        return ', '.join(['{:.1f}% {}'.format(100 * self.values[i], k)
                          for i, k in enumerate(self.keys)])

    def __eq__(self, x):
        """
        Equality
        """
        value = type(self) == type(x)
        if value:
            value &= (self.variables == x.variables)
            value &= (self.probabilities == x.probabilities)
            value &= (self.keys == x.keys)
            value &= (self.values == x.values)
        return value

    def __getitem__(self, event):
        """
        Get value of the probability for some event
        """
        return self.probabilities[event]

    @property
    def value(self):
        """
        This property returns values according to the given probability distribution
        """
        # return np.random.choice(list(self.p.keys()), p=list(self.p.values()))
        return "TODO MAL"

    @property
    def majority_vote(self):
        """
        Return the value with the highest probability
        """
        # return max(self.p.keys(), key=lambda x: self.p[x])
        return self.keys[self.values.index(max(self.values))]

    def marginalize(self, vars_to_marginalize):
        """
        Marginalizes the probability distribution

        Parameters
        ----------
        vars_to_marginalize : list
                              labels of the variables to marginalize/delete

        Returns
        -------
        out : ProbabilityDistribution object
              Marginalized distribution
        """

        # Indices of the variables to marginalize
        indices_to_marginalize = [self.variables.index(v) for v in vars_to_marginalize]
        # Indices and labels of variables that are not deleted
        indices_to_be_kept = [ii for ii in range(len(self.variables))
                              if ii not in indices_to_marginalize]
        vars_to_be_kept = [self.variables[ii] for ii in indices_to_be_kept]

        # Values of the variables
        values_of_all_variables = list(self.keys)
        values_of_vars_to_be_kept = list(set([tuple([x[ii] for ii in indices_to_be_kept])
                                              for x in values_of_all_variables]))
        # Initialize new probability distribution and calculate its values
        pm = {l: 0. for l in values_of_vars_to_be_kept}
        for x, p in self.probabilities.items():
            key = tuple([x[ii] for ii in indices_to_be_kept])
            pm[key] += p

        return ProbabilityDistribution(pm, vars_to_be_kept)

    def get_entropy(self):
        """
        Calculates the entropy of the stored distribution

        Returns
        -------
        h : float
            value of the entropy
        """

        h = 0.
        for x in self.probabilities.values():
            h -= x * np.log2(x + 1e-300)
        return h

    def get_mutual_information(self, labels_x, labels_y):
        """
        Calculates the mutual information of the stored distribution

        Parameters
        ----------
        labels_x, labels_y : list
                             labels of the variables

        Returns
        -------
        i : float
            value of the mutual information
        """

        # Probability distribution for <labels_x>
        vars_to_marginalize_x = [l for l in self.variables if l not in labels_x]
        p_x = self.marginalize(vars_to_marginalize_x)
        # Probability distribution for <labels_y>
        vars_to_marginalize_y = [l for l in self.variables if l not in labels_y]
        p_y = self.marginalize(vars_to_marginalize_y)
        # Probability distribution for both <labels_x> and <labels_y>
        if set(labels_x + labels_y) == set(self.variables):
            p_xy = self
        else:
            vars_to_marginalize_xy = [l for l in self.variables
                                      if not ((l in labels_x) or (l in labels_y))]
            p_xy = self.marginalize(vars_to_marginalize_xy)

        h_x = p_x.get_entropy()
        h_y = p_y.get_entropy()
        h_xy = p_xy.get_entropy()

        return h_x + h_y - h_xy

    def get_conditional_mutual_information(self, labels_x, labels_y, labels_z):
        """
        Calculates the mutual information of the stored distribution

        Parameters
        ----------
        labels_x, labels_y : list
                             labels of the variables
        labels_z : list
                   labels of the conditional variables

        Returns
        -------
        i : float
            value of the mutual information
        """

        # Probability distribution for both <labels_x> and <labels_z>
        vars_to_marginalize_xz = [l for l in self.variables
                                  if not ((l in labels_x) or (l in labels_z))]
        p_xz = self.marginalize(vars_to_marginalize_xz)
        # Probability distribution for both <labels_y> and <labels_z>
        vars_to_marginalize_yz = [l for l in self.variables
                                  if not ((l in labels_y) or (l in labels_z))]
        p_yz = self.marginalize(vars_to_marginalize_yz)
        # Probability distribution for <labels_z>
        vars_to_marginalize_z = [l for l in self.variables if l not in labels_z]
        p_z = self.marginalize(vars_to_marginalize_z)

        # Probability distribution for <labels_x>, <labels_y>, and <labels_z>
        if set(labels_x + labels_y + labels_z) == set(self.variables):
            p_xyz = self
        else:
            vars_to_marginalize_xyz = [l for l in self.variables
                                       if not ((l in labels_x) or (l in labels_y) or
                                               (l in labels_z))]
            p_xyz = self.marginalize(vars_to_marginalize_xyz)

        h_xz = p_xz.get_entropy()
        h_yz = p_yz.get_entropy()
        h_z = p_z.get_entropy()
        h_xyz = p_xyz.get_entropy()

        return h_xz + h_yz - h_xyz - h_z

    def get_kullback_leibler_divergence(self, another_probability_distribution_object):
        """
        Calculated the Kullback-Leibler divergence of this distribution, p, and another one, q.
         That is, KL(q||p) is obtained.

        Parameters
        ----------
        another_probability_distribution_object : ProbabilityDistribution object
                                                  The probability distribution to be compared

        Returns
        -------
        kld : float
              value of the Kullback-Leibler divergence

        """

        if set(another_probability_distribution_object.keys) != set(self.keys):
            raise "The states are different"

        kld = 0.
        for key in self.keys:
            x = self.probabilities[key]
            y = another_probability_distribution_object.probabilities[key]
            kld += y * np.log2((y + 1e-300) / (x + 1e-300))

        return kld
