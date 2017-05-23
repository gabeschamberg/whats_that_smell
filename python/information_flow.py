import os
from jpype import *
import networkx as nx
from nxpd import draw


def compute_te(x, y, kHistory=3, lHistory=3, knns=[8], numSurrogates=0,
                   jarLocation='infodynamics.jar', autoEmbed=True,status=True):

        if jarLocation is None:
            try:
                jarLocation = os.environ['JIDT_PATH'] + '/infodynamics.jar'
            except KeyError:
                print("Must either set jarLocation argument or JIDT_PATH " + \
                      "environment variable to '../infodynamic/infodynamics.jar'")

        # Start the JVM (add the "-Xmx" option with say 1024M if you get crashes due to not enough memory space)
        if not isJVMStarted():
            startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation)
        while not isJVMStarted():
            pass # Wait for JVM to start up

        # Using a KSG estimator for TE is the least biased way to run this:
        teCalcClass = JPackage("infodynamics.measures.continuous.kraskov").TransferEntropyCalculatorKraskov
        teCalc = teCalcClass()
        teCalc.setProperty("NORMALISE", "true")

        teXToY = []
        teYToX = []
        teXToYMean = []
        teXToYStd = []
        teYToXMean = []
        teYToXStd = []

        # Compute a TE value for knn nearest neighbours
        for knn in knns:
            # Perform calculation for X -> Y (lag 1)
            if autoEmbed:
                teCalc.setProperty(teCalcClass.PROP_AUTO_EMBED_METHOD,
                                   teCalcClass.AUTO_EMBED_METHOD_RAGWITZ)
                teCalc.setProperty(teCalcClass.PROP_K_SEARCH_MAX, "10")
                teCalc.setProperty(teCalcClass.PROP_TAU_SEARCH_MAX, "10")
                teCalc.initialise
            else:
                teCalc.initialise(kHistory, 1, lHistory, 1, 1)

            teCalc.setProperty("k", str(knn))
            teCalc.setObservations(JArray(JDouble, 1)(x.tolist()),
                                   JArray(JDouble, 1)(y.tolist()))
            teXToY.append(teCalc.computeAverageLocalOfObservations())
            if numSurrogates > 0:
                teXToYNullDist = teCalc.computeSignificance(numSurrogates)
                teXToYMean.append(teXToYNullDist.getMeanOfDistribution())
                teXToYStd.append(teXToYNullDist.getStdOfDistribution())
                if status:
                    print("X->Y TE with %d NNs: %.4f, with null = %.3f +/- %.3f" %
                      (knn, teXToY[-1], teXToYMean[-1], teXToYStd[-1]))
            else:
                if status:
                    print("X->Y TE with %d NNs: %.4f" %
                      (knn, teXToY[-1]))
            if autoEmbed:
                optimisedK = int(teCalc.getProperty(teCalcClass.K_PROP_NAME))
                optimisedKTau = int(teCalc.getProperty(teCalcClass.K_TAU_PROP_NAME))
                optimisedL = int(teCalc.getProperty(teCalcClass.L_PROP_NAME))
                optimisedLTau = int(teCalc.getProperty(teCalcClass.L_TAU_PROP_NAME))
                if status:
                    print(("X embedding/delay: k=%d, k_tau=%d, Y embedding/delay: l=%d, l_tau=%d") % \
                      (optimisedK, optimisedKTau, optimisedL, optimisedLTau))

            # Perform calculation for Y -> X (lag 1)
            if autoEmbed:
                teCalc.setProperty(teCalcClass.PROP_AUTO_EMBED_METHOD,
                                   teCalcClass.AUTO_EMBED_METHOD_RAGWITZ)
                teCalc.setProperty(teCalcClass.PROP_K_SEARCH_MAX, "10")
                teCalc.setProperty(teCalcClass.PROP_TAU_SEARCH_MAX, "10")
                teCalc.initialise
            else:
                teCalc.initialise(kHistory, 1, lHistory, 1, 1)
                teCalc.setProperty("k", str(knn))

            teCalc.setObservations(JArray(JDouble, 1)(y.tolist()),
                                   JArray(JDouble, 1)(x.tolist()))
            teYToX.append(teCalc.computeAverageLocalOfObservations())
            if numSurrogates > 0:
                teYToXNullDist = teCalc.computeSignificance(numSurrogates)
                teYToXMean.append(teYToXNullDist.getMeanOfDistribution())
                teYToXStd.append(teYToXNullDist.getStdOfDistribution())
                if status:
                    print("Y->X TE with %d NNs: %.4f, with null = %.3f +/- %.3f" %
                      (knn, teYToX[-1], teYToXMean[-1], teYToXStd[-1]))
            else:
                if status:
                    print("Y->X TE with %d NNs: %.4f" %
                      (knn, teYToX[-1]))
            if autoEmbed:
                optimisedK = int(teCalc.getProperty(teCalcClass.K_PROP_NAME))
                optimisedKTau = int(teCalc.getProperty(teCalcClass.K_TAU_PROP_NAME))
                optimisedL = int(teCalc.getProperty(teCalcClass.L_PROP_NAME))
                optimisedLTau = int(teCalc.getProperty(teCalcClass.L_TAU_PROP_NAME))
                if status:
                    print(("Y embedding/delay: k=%d, k_tau=%d, X embedding/delay: l=%d, l_tau=%d") % \
                      (optimisedK, optimisedKTau, optimisedL, optimisedLTau))

        if numSurrogates > 0:
            return teXToY, teYToX, teXToYMean, teXToYStd, teYToXMean, teYToXStd
        else:
            return teXToY, teYToX


def graph_te(tes, labels, colors=None):
    G = nx.DiGraph()
    G.graph['rankdir'] = 'LR'
    G.graph['dpi'] = 120
    # Create nodes
    for label in labels:
        if colors is not None:
            G.add_node(label, style='filled', fillcolor=colors[label])
        else:
            G.add_node(label, style='filled')
    
    for te in tes:
        G.add_edge(labels[te[0]], labels[te[1]], label=str(round(te[2], 4)), penwidth=str(1+10*te[2]**2))
    draw(G, show='ipynb')
    return G