.. code:: ipython3

    import numpy as np
    
    data1 = np.random.normal(0, 1, 1000)
    data2 = np.random.normal(0, 1, size=(2, 1000))
    data3 = np.random.normal(0, 1, size=(4, 1000))

Shannon Entropy
===============

Shannon entropy H is given by the formula
:math:`H=-\sum_{i}p_{i}\log_{b}(p_{i})` where :math:`p_{i}` is the
probability of character number :math:`i` appearing in the stream of
characters of the message.

Consider a simple digital circuit which has a two-bit input (:math:`X`,
:math:`Y`) and a two-bit output (:math:`X` and :math:`Y`, :math:`X` or
:math:`Y`). Assuming that the two input bits :math:`X` and :math:`Y`
have mutually independent chances of :math:`50%` of being *HIGH*, then
the input combinations :math:`(0,0)`, :math:`(0,1)`, :math:`(1,0)`, and
(:math:`1,1)` each have a 1/4 chance of occurring, so the circuit’s
Shannon entropy on the input side is
:math:`H(X,Y)=4{\Big (}-{1 \over 4}\log _{2}{1 \over 4}{\Big )}=2` Then
the possible output combinations are (0,0), (0,1) and (1,1) with
respective chances of 1/4, 1/2, and 1/4 of occurring, so the circuit’s
Shannon entropy on the output side is
:math:`H(X{\text{ and }}Y,X{\text{ or }}Y)=2{\Big (}-{1 \over 4}\log _{2}{1 \over 4}{\Big )}-{1 \over 2}\log _{2}{1 \over 2}=1+{1 \over 2}=1{1 \over 2}`,
so the circuit reduces (or “orders”) the information going through it by
half a bit of Shannon entropy due to its logical irreversibility.

.. code:: ipython3

    from gcpds.entropies import Shannon
    
    ent = Shannon(data1)
    print(f"Input data shape: {data1.shape}")
    print(f"Entropy: {ent}", end='\n\n')


.. parsed-literal::

    Input data shape: (1000,)
    Entropy: 3.4263432580844695
    


.. code:: ipython3

    Shannon(data1, base=10)  # Default base is 2
    Shannon(data1, bins=12)  # Default bins value used to calculate the distribution is 16




.. parsed-literal::

    3.0154281530433003



.. code:: ipython3

    Shannon(data2, conditional=1)




.. parsed-literal::

    3.348718451910984



Joint entropy
-------------

| For 2 variables:
| :math:`{\displaystyle \mathrm {H} (X,Y)=-\sum _{x\in {\mathcal {X}}}\sum _{y\in {\mathcal {Y}}}P(x,y)\log _{2}[P(x,y)]}`

.. code:: ipython3

    ent = Shannon(data2)
    
    print(f"Input data shape: {data2.shape}")
    print(f"Entropy: {ent}")


.. parsed-literal::

    Input data shape: (2, 1000)
    Entropy: 6.454989877719003


| For more than two random variables
  :math:`{\displaystyle X_{1},...,X_{n}} X_{1},...,X_{n}` this expands
  to
| :math:`{\displaystyle \mathrm {H} (X_{1},...,X_{n})=-\sum _{x_{1}\in {\mathcal {X}}_{1}}...\sum _{x_{n}\in {\mathcal {X}}_{n}}P(x_{1},...,x_{n})\log _{2}[P(x_{1},...,x_{n})]}`

.. code:: ipython3

    ent = Shannon(data3)
    
    print(f"Input data shape: {data3.shape}")
    print(f"Entropy: {ent}")


.. parsed-literal::

    Input data shape: (4, 1000)
    Entropy: 9.801254959649105


Conditional entropy
-------------------

| Joint entropy is used in the definition of conditional entropy
| :math:`{\displaystyle \mathrm {H} (X|Y)=\mathrm {H} (X,Y)-\mathrm {H} (Y)}`

.. code:: ipython3

    ent = Shannon(data3, conditional=0)  # `conditional` is an index of the input array
    
    print(f"Input data shape: {data3.shape}")
    print(f"Entropy: {ent}")


.. parsed-literal::

    Input data shape: (4, 1000)
    Entropy: 6.332143741478394


--------------

References
~~~~~~~~~~

-  Thomas M. Cover; Joy A. Thomas. Elements of Information Theory.
   Hoboken, New Jersey: Wiley. ISBN 0-471-24195-4.
