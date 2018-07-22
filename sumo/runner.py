#!/usr/bin/env python
# Eclipse SUMO, Simulation of Urban MObility; see https://eclipse.org/sumo
# Copyright (C) 2009-2017 German Aerospace Center (DLR) and others.
# This program and the accompanying materials
# are made available under the terms of the Eclipse Public License v2.0
# which accompanies this distribution, and is available at
# http://www.eclipse.org/legal/epl-v20.html

# @file    runner.py
# @author  Lena Kalleske
# @author  Daniel Krajzewicz
# @author  Michael Behrisch
# @author  Jakob Erdmann
# @date    2009-03-26
# @version $Id$
#
# dependence on a release of sumo https://github.com/eclipse/sumo https://www.dlr.de/ts/en/Portaldata/16/Resources/projekte/sumo/sumo-win64-0.32.0.msi
# sumolib https://github.com/eclipse/sumo/tree/master/tools/sumolib
# @author  Yun
# @data    2018-05-17

from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import optparse
import subprocess
import random
import numpy as np

# we need to import python modules from the $SUMO_HOME    ools directory
try:
    sys.path.append(os.path.join(os.path.dirname(
        __file__), '..', '..', '..', '..', "tools"))  # tutorial in tests
    sys.path.append(os.path.join(os.environ.get("SUMO_HOME", os.path.join(
        os.path.dirname(__file__), "..", "..", "..")), "tools"))  # tutorial in docs
    from sumolib import checkBinary  # noqa
except ImportError:
    sys.exit(
        "please declare environment variable 'SUMO_HOME' as the root directory of your sumo installation (it should contain folders 'bin', 'tools' and 'docs')")

import traci


def generate_routefile(flow, signal):
    random.seed(42)  # make tests reproducible
    # N = 3600  # number of time steps
    N = int(signal["cycle"])
    # demand per second from different directions
    p1 = flow["65-1-57-3"] / (N + 1)
    p2 = flow["65-1-66-1"] / (N + 1)
    p3 = flow["65-5-83-7"] / (N + 1)
    p4 = flow["65-5-64-5"] / (N + 1)
    p5 = flow["65-3-57-3"] / (N + 1)
    p6 = flow["65-7-83-7"] / (N + 1)
    with open("65.rou.xml", "w", encoding="utf-8") as routes:
        print("""<routes>
        <vType id="passenger65-1-57-3" speedFactor="normc(1.00,0.10,0.20,2.00)" vClass="passenger"/>
        <vType id="passenger65-1-66-1" speedFactor="normc(1.00,0.10,0.20,2.00)" vClass="passenger"/>
        <vType id="passenger65-5-83-7" speedFactor="normc(1.00,0.10,0.20,2.00)" vClass="passenger"/>
        <vType id="passenger65-5-64-5" speedFactor="normc(1.00,0.10,0.20,2.00)" vClass="passenger"/>
        <vType id="passenger65-3-57-3" speedFactor="normc(1.00,0.10,0.20,2.00)" vClass="passenger"/>
        <vType id="passenger65-7-83-7" speedFactor="normc(1.00,0.10,0.20,2.00)" vClass="passenger"/>
        <route id="65-1-57-3" edges="65-1 57-3" />
        <route id="65-1-66-1" edges="65-1 66-1" />
        <route id="65-5-83-7" edges="65-5 83-7" />
        <route id="65-5-64-5" edges="65-5 64-5" />
        <route id="65-3-57-3" edges="65-3 57-3" />
        <route id="65-7-83-7" edges="65-7 83-7" />
        """, file=routes)
        lastVeh = 0
        vehNr = 0
        for i in range(N):
            if random.uniform(0, 1) < p1:
                print('    <vehicle id="65-1-57-3_%i" type="passenger65-1-57-3" route="65-1-57-3" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
                lastVeh = i
            if random.uniform(0, 1) < p2:
                print('    <vehicle id="65-1-66-1_%i" type="passenger65-1-66-1" route="65-1-66-1" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
                lastVeh = i
            if random.uniform(0, 1) < p3:
                print('    <vehicle id="65-5-83-7_%i" type="passenger65-5-83-7" route="65-5-83-7" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
                lastVeh = i
            if random.uniform(0, 1) < p4:
                print('    <vehicle id="65-5-64-5_%i" type="passenger65-5-64-5" route="65-5-64-5" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
                lastVeh = i
            if random.uniform(0, 1) < p5:
                print('    <vehicle id="65-3-57-3_%i" type="passenger65-3-57-3" route="65-3-57-3" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
                lastVeh = i
            if random.uniform(0, 1) < p6:
                print('    <vehicle id="65-7-83-7_%i" type="passenger65-7-83-7" route="65-7-83-7" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
                lastVeh = i

        print("</routes>", file=routes)


def run():
    """execute the TraCI control loop"""
    step = 0
    # we start with phase 2 where EW has green
    traci.trafficlight.setPhase("65", 2)
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        step += 1
    traci.close()
    sys.stdout.flush()


import sumolib.output as output


def output_results():
    delay1 = []
    delay2 = []
    delay3 = []
    delay4 = []
    delay5 = []
    delay6 = []
    for trip in output.parse('tripinfo.xml', ['tripinfo']):
        if (trip.vType == "passenger65-1-57-3"):
            delay1.append(trip.timeLoss)
        if (trip.vType == "passenger65-1-66-1"):
            delay2.append(trip.timeLoss)
        if (trip.vType == "passenger65-5-83-7"):
            delay3.append(trip.timeLoss)
        if (trip.vType == "passenger65-5-64-5"):
            delay4.append(trip.timeLoss)
        if (trip.vType == "passenger65-3-57-3"):
            delay5.append(trip.timeLoss)
        if (trip.vType == "passenger65-7-83-7"):
            delay6.append(trip.timeLoss)
    return np.array([np.array(delay1, dtype='float32'),
                     np.array(delay2, dtype='float32'),
                     np.array(delay3, dtype='float32'),
                     np.array(delay4, dtype='float32'),
                     np.array(delay5, dtype='float32'),
                     np.array(delay6, dtype='float32')])


def generate_netfile(signal):
    with open("65.net.xml", "w", encoding="utf-8") as net:
        print("""<?xml version="1.0" encoding="UTF-8"?>
<net version="0.27" junctionCornerDetail="5" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/net_file.xsd">
    <location netOffset="-279952.52,-3467712.15" convBoundary="0.00,0.00,10795.30,1349.83" origBoundary="120.687371,31.314441,120.800578,31.341757" projParameter="+proj=utm +zone=51 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"/>
    <type id="highway.bridleway" priority="1" numLanes="1" speed="2.78" allow="pedestrian" oneway="1" width="2.00"/>
    <type id="highway.bus_guideway" priority="1" numLanes="1" speed="8.33" allow="bus" oneway="1"/>
    <type id="highway.cycleway" priority="1" numLanes="1" speed="5.56" allow="bicycle" oneway="0" width="1.00"/>
    <type id="highway.footway" priority="1" numLanes="1" speed="2.78" allow="pedestrian" oneway="1" width="2.00"/>
    <type id="highway.ford" priority="1" numLanes="1" speed="2.78" allow="army" oneway="0"/>
    <type id="highway.living_street" priority="3" numLanes="1" speed="2.78" disallow="tram rail_urban rail rail_electric ship" oneway="0"/>
    <type id="highway.motorway" priority="13" numLanes="2" speed="44.44" allow="private emergency authority army vip passenger hov taxi bus coach delivery truck trailer motorcycle evehicle custom1 custom2" oneway="1"/>
    <type id="highway.motorway_link" priority="12" numLanes="1" speed="22.22" allow="private emergency authority army vip passenger hov taxi bus coach delivery truck trailer motorcycle evehicle custom1 custom2" oneway="1"/>
    <type id="highway.path" priority="1" numLanes="1" speed="2.78" allow="bicycle pedestrian" oneway="1" width="2.00"/>
    <type id="highway.pedestrian" priority="1" numLanes="1" speed="2.78" allow="pedestrian" oneway="1" width="2.00"/>
    <type id="highway.primary" priority="9" numLanes="2" speed="27.78" disallow="tram rail_urban rail rail_electric ship" oneway="0"/>
    <type id="highway.primary_link" priority="8" numLanes="1" speed="22.22" disallow="tram rail_urban rail rail_electric ship" oneway="0"/>
    <type id="highway.raceway" priority="14" numLanes="2" speed="83.33" allow="vip" oneway="0"/>
    <type id="highway.residential" priority="4" numLanes="1" speed="13.89" disallow="tram rail_urban rail rail_electric ship" oneway="0"/>
    <type id="highway.secondary" priority="7" numLanes="2" speed="27.78" disallow="tram rail_urban rail rail_electric ship" oneway="0"/>
    <type id="highway.secondary_link" priority="6" numLanes="1" speed="22.22" disallow="tram rail_urban rail rail_electric ship" oneway="0"/>
    <type id="highway.service" priority="2" numLanes="1" speed="5.56" allow="delivery bicycle pedestrian" oneway="0"/>
    <type id="highway.services" priority="1" numLanes="1" speed="8.33" disallow="tram rail_urban rail rail_electric ship" oneway="0"/>
    <type id="highway.stairs" priority="1" numLanes="1" speed="1.39" allow="pedestrian" oneway="1" width="2.00"/>
    <type id="highway.step" priority="1" numLanes="1" speed="1.39" allow="pedestrian" oneway="1" width="2.00"/>
    <type id="highway.steps" priority="1" numLanes="1" speed="1.39" allow="pedestrian" oneway="1" width="2.00"/>
    <type id="highway.tertiary" priority="6" numLanes="1" speed="22.22" disallow="tram rail_urban rail rail_electric ship" oneway="0"/>
    <type id="highway.tertiary_link" priority="5" numLanes="1" speed="22.22" disallow="tram rail_urban rail rail_electric ship" oneway="0"/>
    <type id="highway.track" priority="1" numLanes="1" speed="5.56" disallow="tram rail_urban rail rail_electric ship" oneway="0"/>
    <type id="highway.trunk" priority="11" numLanes="2" speed="27.78" disallow="tram rail_urban rail rail_electric bicycle pedestrian ship" oneway="0"/>
    <type id="highway.trunk_link" priority="10" numLanes="1" speed="22.22" disallow="tram rail_urban rail rail_electric bicycle pedestrian ship" oneway="0"/>
    <type id="highway.unclassified" priority="5" numLanes="1" speed="13.89" disallow="tram rail_urban rail rail_electric ship" oneway="0"/>
    <type id="highway.unsurfaced" priority="1" numLanes="1" speed="8.33" disallow="tram rail_urban rail rail_electric ship" oneway="0"/>
    <type id="railway.light_rail" priority="15" numLanes="1" speed="27.78" allow="rail_urban" oneway="1"/>
    <type id="railway.preserved" priority="15" numLanes="1" speed="27.78" allow="rail" oneway="1"/>
    <type id="railway.rail" priority="15" numLanes="1" speed="83.33" allow="rail rail_electric" oneway="1"/>
    <type id="railway.subway" priority="15" numLanes="1" speed="27.78" allow="rail_urban" oneway="1"/>
    <type id="railway.tram" priority="15" numLanes="1" speed="13.89" allow="tram" oneway="1"/>
    <edge id=":65_0" function="internal">
        <lane id=":65_0_0" index="0" disallow="tram rail_urban rail rail_electric ship" speed="27.78" length="6.18" shape="2737.68,455.10 2735.77,455.24 2734.37,455.79 2733.49,456.77 2733.13,458.17"/>
    </edge>
    <edge id=":65_1" function="internal">
        <lane id=":65_1_0" index="0" disallow="tram rail_urban rail rail_electric ship" speed="27.78" length="28.68" shape="2737.68,455.10 2709.15,453.62"/>
        <lane id=":65_1_1" index="1" disallow="tram rail_urban rail rail_electric ship" speed="27.78" length="28.68" shape="2737.80,451.80 2709.19,450.32"/>
        <lane id=":65_1_2" index="2" disallow="tram rail_urban rail rail_electric ship" speed="27.78" length="28.68" shape="2737.92,448.50 2709.24,447.02"/>
        <lane id=":65_1_3" index="3" disallow="tram rail_urban rail rail_electric ship" speed="27.78" length="28.68" shape="2738.04,445.20 2709.29,443.72"/>
    </edge>
    <edge id=":65_5" function="internal">
        <lane id=":65_5_0" index="0" disallow="tram rail_urban rail rail_electric ship" speed="27.78" length="17.03" shape="2738.04,445.20 2728.92,443.57 2722.69,439.37 2722.58,439.15"/>
    </edge>
    <edge id=":65_6" function="internal">
        <lane id=":65_6_0" index="0" disallow="tram rail_urban rail rail_electric ship" speed="27.78" length="7.58" shape="2738.04,445.20 2735.41,443.31 2734.58,441.48 2735.54,439.72 2735.82,439.55"/>
    </edge>
    <edge id=":65_24" function="internal">
        <lane id=":65_24_0" index="0" speed="27.78" length="16.66" shape="2722.58,439.15 2719.35,432.60 2718.91,423.25"/>
    </edge>
    <edge id=":65_25" function="internal">
        <lane id=":65_25_0" index="0" speed="27.78" length="2.91" shape="2735.82,439.55 2738.30,438.02"/>
    </edge>
    <edge id=":65_7" function="internal">
        <lane id=":65_7_0" index="0" disallow="tram rail_urban rail rail_electric ship" speed="27.78" length="5.07" shape="2735.76,424.83 2735.83,426.23 2736.33,427.25 2737.27,427.88 2738.65,428.13"/>
    </edge>
    <edge id=":65_8" function="internal">
        <lane id=":65_8_0" index="0" disallow="tram rail_urban rail rail_electric ship" speed="27.78" length="33.49" shape="2735.76,424.83 2733.13,458.17"/>
        <lane id=":65_8_1" index="1" disallow="tram rail_urban rail rail_electric ship" speed="27.78" length="33.49" shape="2732.47,424.52 2729.84,457.96"/>
    </edge>
    <edge id=":65_10" function="internal">
        <lane id=":65_10_0" index="0" disallow="tram rail_urban rail rail_electric ship" speed="27.78" length="24.45" shape="2732.47,424.52 2730.33,433.04 2725.76,439.07 2718.74,442.63 2718.51,442.66"/>
    </edge>
    <edge id=":65_11" function="internal">
        <lane id=":65_11_0" index="0" disallow="tram rail_urban rail rail_electric ship" speed="27.78" length="17.05" shape="2732.47,424.52 2728.60,429.29 2725.05,430.67 2721.82,428.66 2720.26,425.76"/>
    </edge>
    <edge id=":65_26" function="internal">
        <lane id=":65_26_0" index="0" speed="27.78" length="9.29" shape="2718.51,442.66 2709.29,443.72"/>
    </edge>
    <edge id=":65_27" function="internal">
        <lane id=":65_27_0" index="0" speed="27.78" length="2.84" shape="2720.26,425.76 2718.91,423.25"/>
    </edge>
    <edge id=":65_12" function="internal">
        <lane id=":65_12_0" index="0" disallow="tram rail_urban rail rail_electric ship" speed="27.78" length="8.34" shape="2709.52,427.25 2712.04,427.01 2713.89,426.22 2715.09,424.86 2715.62,422.94"/>
    </edge>
    <edge id=":65_13" function="internal">
        <lane id=":65_13_0" index="0" disallow="tram rail_urban rail rail_electric ship" speed="27.78" length="29.03" shape="2709.52,427.25 2738.65,428.13"/>
        <lane id=":65_13_1" index="1" disallow="tram rail_urban rail rail_electric ship" speed="27.78" length="29.03" shape="2709.48,430.55 2738.53,431.43"/>
        <lane id=":65_13_2" index="2" disallow="tram rail_urban rail rail_electric ship" speed="27.78" length="29.03" shape="2709.43,433.85 2738.41,434.72"/>
        <lane id=":65_13_3" index="3" disallow="tram rail_urban rail rail_electric ship" speed="27.78" length="29.03" shape="2709.38,437.15 2738.30,438.02"/>
    </edge>
    <edge id=":65_17" function="internal">
        <lane id=":65_17_0" index="0" disallow="tram rail_urban rail rail_electric ship" speed="27.78" length="17.69" shape="2709.38,437.15 2718.82,438.58 2725.37,442.53 2725.62,442.96"/>
    </edge>
    <edge id=":65_18" function="internal">
        <lane id=":65_18_0" index="0" disallow="tram rail_urban rail rail_electric ship" speed="27.78" length="6.87" shape="2709.38,437.15 2711.82,438.82 2712.62,440.48 2711.78,442.11 2711.58,442.23"/>
    </edge>
    <edge id=":65_28" function="internal">
        <lane id=":65_28_0" index="0" speed="27.78" length="15.94" shape="2725.62,442.96 2729.05,448.99 2729.84,457.96"/>
    </edge>
    <edge id=":65_29" function="internal">
        <lane id=":65_29_0" index="0" speed="27.78" length="2.73" shape="2711.58,442.23 2709.29,443.72"/>
    </edge>
    <edge id=":65_19" function="internal">
        <lane id=":65_19_0" index="0" disallow="tram rail_urban rail rail_electric ship" speed="27.78" length="5.06" shape="2712.12,456.84 2712.01,455.44 2711.48,454.44 2710.52,453.84 2709.15,453.62"/>
    </edge>
    <edge id=":65_20" function="internal">
        <lane id=":65_20_0" index="0" disallow="tram rail_urban rail rail_electric ship" speed="27.78" length="34.02" shape="2712.12,456.84 2715.62,422.94"/>
        <lane id=":65_20_1" index="1" disallow="tram rail_urban rail rail_electric ship" speed="27.78" length="34.02" shape="2715.42,457.05 2718.91,423.25"/>
    </edge>
    <edge id=":65_22" function="internal">
        <lane id=":65_22_0" index="0" disallow="tram rail_urban rail rail_electric ship" speed="27.78" length="25.12" shape="2715.42,457.05 2717.32,448.43 2721.76,442.39 2728.76,438.92 2729.74,438.83"/>
    </edge>
    <edge id=":65_23" function="internal">
        <lane id=":65_23_0" index="0" disallow="tram rail_urban rail rail_electric ship" speed="27.78" length="18.28" shape="2715.42,457.05 2719.37,451.87 2723.09,450.29 2726.58,452.32 2728.43,455.52"/>
    </edge>
    <edge id=":65_30" function="internal">
        <lane id=":65_30_0" index="0" speed="27.78" length="8.59" shape="2729.74,438.83 2738.30,438.02"/>
    </edge>
    <edge id=":65_31" function="internal">
        <lane id=":65_31_0" index="0" speed="27.78" length="2.82" shape="2728.43,455.52 2729.84,457.96"/>
    </edge>
    <edge id="57-3" from="65" to="57o" name="现代大道" priority="9" type="highway.primary" spreadType="center" shape="2733.34,432.89 3291.36,453.01 3300.91,453.09">
        <lane id="57-3_0" index="0" disallow="tram rail_urban rail rail_electric ship" speed="27.78" length="562.80" shape="2738.65,428.13 3291.47,448.06 3300.95,448.14">
            <param key="origId" value="148708441"/>
        </lane>
        <lane id="57-3_1" index="1" disallow="tram rail_urban rail rail_electric ship" speed="27.78" length="562.80" shape="2738.53,431.43 3291.40,451.36 3300.92,451.44">
            <param key="origId" value="148708441"/>
        </lane>
        <lane id="57-3_2" index="2" disallow="tram rail_urban rail rail_electric ship" speed="27.78" length="562.80" shape="2738.41,434.72 3291.32,454.66 3300.90,454.74">
            <param key="origId" value="148708441"/>
        </lane>
        <lane id="57-3_3" index="3" disallow="tram rail_urban rail rail_electric ship" speed="27.78" length="562.80" shape="2738.30,438.02 3291.25,457.96 3300.87,458.04">
            <param key="origId" value="148708441"/>
        </lane>
    </edge>
    <edge id="64-5" from="65" to="64o" name="星湖街" priority="9" type="highway.primary" spreadType="center" shape="2732.00,449.96 2717.90,672.17 2688.75,1078.66 2688.12,1089.36 2672.47,1349.83">
        <lane id="64-5_0" index="0" disallow="tram rail_urban rail rail_electric ship" speed="27.78" length="893.73" shape="2733.13,458.17 2719.55,672.28 2690.40,1078.77 2689.77,1089.46 2674.12,1349.93">
            <param key="origId" value="350701826"/>
        </lane>
        <lane id="64-5_1" index="1" disallow="tram rail_urban rail rail_electric ship" speed="27.78" length="893.73" shape="2729.84,457.96 2716.25,672.06 2687.10,1078.55 2686.47,1089.26 2670.82,1349.73">
            <param key="origId" value="350701826"/>
        </lane>
    </edge>
    <edge id="65-1" from="64i" to="65" name="星湖街" priority="9" type="highway.primary" spreadType="center" shape="2654.85,1346.92 2670.62,1087.70 2671.21,1076.83 2700.37,668.55 2714.29,448.74">
        <lane id="65-1_0" index="0" disallow="tram rail_urban rail rail_electric ship" speed="27.78" length="891.94" shape="2653.20,1346.82 2668.97,1087.60 2669.56,1076.73 2698.72,668.44 2712.12,456.84">
            <param key="origId" value="148708414"/>
        </lane>
        <lane id="65-1_1" index="1" disallow="tram rail_urban rail rail_electric ship" speed="27.78" length="891.94" shape="2656.50,1347.02 2672.27,1087.80 2672.86,1076.93 2702.02,668.66 2715.42,457.05">
            <param key="origId" value="148708414"/>
        </lane>
    </edge>
    <edge id="65-3" from="265588904" to="65" name="现代大道" priority="9" type="highway.primary" spreadType="center" shape="1993.58,474.25 2113.66,458.50 2236.85,444.76 2299.68,439.05 2364.46,434.52 2426.99,432.35 2430.86,432.21 2562.28,429.78 2716.40,432.31">
        <lane id="65-3_0" index="0" disallow="tram rail_urban rail rail_electric ship" speed="27.78" length="718.16" shape="1992.94,469.34 2113.06,453.59 2236.35,439.83 2299.28,434.12 2364.20,429.58 2426.82,427.40 2430.72,427.26 2562.27,424.83 2709.52,427.25">
            <param key="origId" value="148708441"/>
        </lane>
        <lane id="65-3_1" index="1" disallow="tram rail_urban rail rail_electric ship" speed="27.78" length="718.16" shape="1993.37,472.61 2113.46,456.86 2236.68,443.12 2299.55,437.41 2364.37,432.87 2426.93,430.70 2430.81,430.56 2562.28,428.13 2709.48,430.55">
            <param key="origId" value="148708441"/>
        </lane>
        <lane id="65-3_2" index="2" disallow="tram rail_urban rail rail_electric ship" speed="27.78" length="718.16" shape="1993.79,475.89 2113.86,460.14 2237.02,446.40 2299.81,440.69 2364.55,436.17 2427.05,434.00 2430.91,433.86 2562.28,431.43 2709.43,433.85">
            <param key="origId" value="148708441"/>
        </lane>
        <lane id="65-3_3" index="3" disallow="tram rail_urban rail rail_electric ship" speed="27.78" length="718.16" shape="1994.22,479.16 2114.26,463.41 2237.35,449.69 2300.08,443.98 2364.72,439.46 2427.16,437.30 2431.00,437.16 2562.29,434.73 2709.38,437.15">
            <param key="origId" value="148708441"/>
        </lane>
    </edge>
    <edge id="65-5" from="66i" to="65" name="星湖街" priority="9" type="highway.primary" spreadType="center" shape="2774.86,2.89 2772.06,28.22 2769.10,70.46 2763.21,118.62 2757.56,178.48 2756.25,190.35 2733.34,432.89">
        <lane id="65-5_0" index="0" disallow="tram rail_urban rail rail_electric ship" speed="27.78" length="423.79" shape="2776.50,3.07 2773.70,28.37 2770.74,70.62 2764.85,118.80 2759.20,178.65 2757.89,190.52 2735.76,424.83">
            <param key="origId" value="350701826"/>
        </lane>
        <lane id="65-5_1" index="1" disallow="tram rail_urban rail rail_electric ship" speed="27.78" length="423.79" shape="2773.22,2.71 2770.42,28.07 2767.46,70.30 2761.57,118.44 2755.92,178.31 2754.61,190.18 2732.47,424.52">
            <param key="origId" value="350701826"/>
        </lane>
    </edge>
    <edge id="65-7" from="57i" to="65" name="现代大道" priority="9" type="highway.primary" spreadType="center" shape="3299.47,468.25 2732.00,449.96">
        <lane id="65-7_0" index="0" disallow="tram rail_urban rail rail_electric ship" speed="27.78" length="561.90" shape="3299.31,473.20 2737.68,455.10">
            <param key="origId" value="483927083"/>
        </lane>
        <lane id="65-7_1" index="1" disallow="tram rail_urban rail rail_electric ship" speed="27.78" length="561.90" shape="3299.42,469.90 2737.80,451.80">
            <param key="origId" value="483927083"/>
        </lane>
        <lane id="65-7_2" index="2" disallow="tram rail_urban rail rail_electric ship" speed="27.78" length="561.90" shape="3299.52,466.60 2737.92,448.50">
            <param key="origId" value="483927083"/>
        </lane>
        <lane id="65-7_3" index="3" disallow="tram rail_urban rail rail_electric ship" speed="27.78" length="561.90" shape="3299.63,463.30 2738.04,445.20">
            <param key="origId" value="483927083"/>
        </lane>
    </edge>
    <edge id="66-1" from="65" to="66o" name="星湖街" priority="9" type="highway.primary" spreadType="center" shape="2716.40,432.31 2739.28,188.33 2740.57,176.44 2745.78,115.93 2750.79,69.62 2753.36,49.96 2756.98,27.76 2761.85,0.00">
        <lane id="66-1_0" index="0" disallow="tram rail_urban rail rail_electric ship" speed="27.78" length="425.58" shape="2715.62,422.94 2737.64,188.16 2738.93,176.28 2744.14,115.77 2749.15,69.42 2751.73,49.72 2755.35,27.48 2760.22,-0.29">
            <param key="origId" value="148708414"/>
        </lane>
        <lane id="66-1_1" index="1" disallow="tram rail_urban rail rail_electric ship" speed="27.78" length="425.58" shape="2718.91,423.25 2740.92,188.50 2742.21,176.60 2747.42,116.09 2752.43,69.82 2754.99,50.20 2758.61,28.04 2763.48,0.29">
            <param key="origId" value="148708414"/>
        </lane>
    </edge>
    <edge id="83-7" from="65" to="3702401796" name="现代大道" priority="9" type="highway.primary" spreadType="center" shape="2714.29,448.74 2562.09,446.56 2427.26,447.99 2364.79,450.66 2300.43,455.51 2238.86,461.41 2148.20,471.33 1993.40,489.99">
        <lane id="83-7_0" index="0" disallow="tram rail_urban rail rail_electric ship" speed="27.78" length="718.02" shape="2709.15,453.62 2562.08,451.51 2427.39,452.94 2365.08,455.60 2300.85,460.44 2239.37,466.33 2148.77,476.25 1993.99,494.90">
            <param key="origId" value="483927083"/>
        </lane>
        <lane id="83-7_1" index="1" disallow="tram rail_urban rail rail_electric ship" speed="27.78" length="718.02" shape="2709.19,450.32 2562.09,448.21 2427.30,449.64 2364.89,452.31 2300.57,457.15 2239.03,463.05 2148.39,472.97 1993.60,491.63">
            <param key="origId" value="483927083"/>
        </lane>
        <lane id="83-7_2" index="2" disallow="tram rail_urban rail rail_electric ship" speed="27.78" length="718.02" shape="2709.24,447.02 2562.09,444.91 2427.22,446.34 2364.69,449.01 2300.29,453.87 2238.69,459.77 2148.01,469.69 1993.20,488.35">
            <param key="origId" value="483927083"/>
        </lane>
        <lane id="83-7_3" index="3" disallow="tram rail_urban rail rail_electric ship" speed="27.78" length="718.02" shape="2709.29,443.72 2562.10,441.61 2427.13,443.04 2364.50,445.72 2300.01,450.58 2238.35,456.49 2147.63,466.41 1992.81,485.08">
            <param key="origId" value="483927083"/>
        </lane>
    </edge>""", file=net)
        print("""<tlLogic id="65" type="static" programID="65" offset="0">""", file=net)
        print('<phase duration="' + str(signal["65-1-57-3"]) + '" state="GGGGGGGrrrrrrrrrrrrrrrrr"' + "/>", file=net)
        print('<phase duration="' + str(signal["65-5-83-7"]) + '" state="rrrrrrrGGGGGrrrrrrrrrrrr"' + "/>", file=net)
        print('<phase duration="' + str(signal["65-5-64-5"]) + '" state="rrrrrrrrrrrrGGGGGGGrrrrr"' + "/>", file=net)
        print('<phase duration="' + str(signal["65-3-57-3"]) + '" state="rrrrrrrrrrrrrrrrrrrGGGGG"' + "/>", file=net)
        print("</tlLogic>", file=net)
        print("""    <junction id="265588904" type="dead_end" x="3.21" y="455.71" incLanes="" intLanes="" shape=""/>
    <junction id="3702401796" type="dead_end" x="0.00" y="476.29" incLanes="83-7_0 83-7_1 83-7_2 83-7_3" intLanes="" shape=""/>
    <junction id="57i" type="dead_end" x="9957.69" y="1180.05" incLanes="" intLanes="" shape=""/>
    <junction id="57o" type="dead_end" x="10795.30" y="1315.84" incLanes="57-3_0 57-3_1 57-3_2 57-3_3" intLanes="" shape=""/>
    <junction id="64i" type="dead_end" x="2654.85" y="1346.92" incLanes="" intLanes="" shape="2658.09,1347.12 2651.61,1346.72"/>
    <junction id="64o" type="dead_end" x="2672.47" y="1349.83" incLanes="64-5_0 64-5_1" intLanes="" shape="2675.71,1350.02 2669.23,1349.64"/>
    <junction id="65" type="traffic_light" x="2724.01" y="440.98" incLanes="65-7_0 65-7_1 65-7_2 65-7_3 65-5_0 65-5_1 65-3_0 65-3_1 65-3_2 65-3_3 65-1_0 65-1_1" intLanes=":65_0_0 :65_1_0 :65_1_1 :65_1_2 :65_1_3 :65_24_0 :65_25_0 :65_7_0 :65_8_0 :65_8_1 :65_26_0 :65_27_0 :65_12_0 :65_13_0 :65_13_1 :65_13_2 :65_13_3 :65_28_0 :65_29_0 :65_19_0 :65_20_0 :65_20_1 :65_30_0 :65_31_0" shape="2737.62,456.69 2738.71,426.53 2737.89,426.33 2737.62,426.12 2737.44,425.82 2737.35,425.44 2737.35,424.98 2714.03,422.79 2713.40,424.41 2712.76,424.97 2711.91,425.36 2710.83,425.59 2709.55,425.65 2709.12,455.22 2709.95,455.36 2710.23,455.57 2710.42,455.86 2710.52,456.25 2710.53,456.74 2734.73,458.27 2735.10,457.36 2735.51,457.04 2736.06,456.83 2736.77,456.71">
        <request index="0" response="000000000000000000000000" foes="100000100000001100000000" cont="0"/>
        <request index="1" response="000000000000000000000000" foes="011111100000011100000000" cont="0"/>
        <request index="2" response="000000000000000000000000" foes="011111100000011100000000" cont="0"/>
        <request index="3" response="000000000000000000000000" foes="011111100000011100000000" cont="0"/>
        <request index="4" response="000000000000000000000000" foes="011111100000011100000000" cont="0"/>
        <request index="5" response="000000011111000000000000" foes="011100011111111100000000" cont="1"/>
        <request index="6" response="010000011110000010000000" foes="010000011110000010000000" cont="1"/>
        <request index="7" response="000000011110000000000000" foes="010000011110000001000000" cont="0"/>
        <request index="8" response="000000111110000000111111" foes="110000111110000000111111" cont="0"/>
        <request index="9" response="000000111110000000111111" foes="110000111110000000111111" cont="0"/>
        <request index="10" response="001110111110000000111110" foes="001111111110000000111110" cont="1"/>
        <request index="11" response="001100000001000000100000" foes="001100000001000000100000" cont="1"/>
        <request index="12" response="000000000000000000000000" foes="001100000000100000100000" cont="0"/>
        <request index="13" response="000000000000000000000000" foes="011100000000011111100000" cont="0"/>
        <request index="14" response="000000000000000000000000" foes="011100000000011111100000" cont="0"/>
        <request index="15" response="000000000000000000000000" foes="011100000000011111100000" cont="0"/>
        <request index="16" response="000000000000000000000000" foes="011100000000011111100000" cont="0"/>
        <request index="17" response="000000000000000000011111" foes="111100000000011100011111" cont="1"/>
        <request index="18" response="000010000000010000011110" foes="000010000000010000011110" cont="1"/>
        <request index="19" response="000000000000000000011110" foes="000001000000010000011110" cont="0"/>
        <request index="20" response="000000111111000000111110" foes="000000111111110000111110" cont="0"/>
        <request index="21" response="000000111111000000111110" foes="000000111111110000111110" cont="0"/>
        <request index="22" response="000000111110001110111110" foes="000000111110001111111110" cont="1"/>
        <request index="23" response="000000100000001100000001" foes="000000100000001100000001" cont="1"/>
    </junction>
    <junction id="66i" type="dead_end" x="2774.86" y="2.89" incLanes="" intLanes="" shape="2771.63,2.53 2778.09,3.25"/>
    <junction id="66o" type="dead_end" x="2761.85" y="0.00" incLanes="66-1_0 66-1_1" intLanes="" shape="2758.65,-0.56 2765.05,0.56"/>
    <junction id=":65_24_0" type="internal" x="2722.58" y="439.15" incLanes=":65_5_0 65-3_0 65-3_1 65-3_2 65-3_3" intLanes=":65_8_0 :65_8_1 :65_10_0 :65_11_0 :65_12_0 :65_13_0 :65_13_1 :65_13_2 :65_13_3 :65_20_0 :65_20_1 :65_22_0"/>
    <junction id=":65_25_0" type="internal" x="2735.82" y="439.55" incLanes=":65_6_0 65-1_1 65-3_0 65-3_1 65-3_2 65-3_3 65-5_0" intLanes=":65_7_0 :65_13_0 :65_13_1 :65_13_2 :65_13_3 :65_22_0"/>
    <junction id=":65_26_0" type="internal" x="2718.51" y="442.66" incLanes=":65_10_0 65-1_0 65-1_1" intLanes=":65_1_0 :65_1_1 :65_1_2 :65_1_3 :65_5_0 :65_13_0 :65_13_1 :65_13_2 :65_13_3 :65_17_0 :65_18_0 :65_19_0 :65_20_0 :65_20_1"/>
    <junction id=":65_27_0" type="internal" x="2720.26" y="425.76" incLanes=":65_11_0 65-1_0 65-1_1 65-3_0 65-7_3" intLanes=":65_5_0 :65_12_0 :65_20_0 :65_20_1"/>
    <junction id=":65_28_0" type="internal" x="2725.62" y="442.96" incLanes=":65_17_0 65-7_0 65-7_1 65-7_2 65-7_3" intLanes=":65_0_0 :65_1_0 :65_1_1 :65_1_2 :65_1_3 :65_8_0 :65_8_1 :65_10_0 :65_20_0 :65_20_1 :65_22_0 :65_23_0"/>
    <junction id=":65_29_0" type="internal" x="2711.58" y="442.23" incLanes=":65_18_0 65-1_0 65-5_1 65-7_0 65-7_1 65-7_2 65-7_3" intLanes=":65_1_0 :65_1_1 :65_1_2 :65_1_3 :65_10_0 :65_19_0"/>
    <junction id=":65_30_0" type="internal" x="2729.74" y="438.83" incLanes=":65_22_0 65-5_0 65-5_1" intLanes=":65_1_0 :65_1_1 :65_1_2 :65_1_3 :65_5_0 :65_6_0 :65_7_0 :65_8_0 :65_8_1 :65_13_0 :65_13_1 :65_13_2 :65_13_3 :65_17_0"/>
    <junction id=":65_31_0" type="internal" x="2728.43" y="455.52" incLanes=":65_23_0 65-3_3 65-5_0 65-5_1 65-7_0" intLanes=":65_0_0 :65_8_0 :65_8_1 :65_17_0"/>
    <connection from="65-1" to="83-7" fromLane="0" toLane="0" via=":65_19_0" tl="65" linkIndex="19" dir="r" state="o"/>
    <connection from="65-1" to="66-1" fromLane="0" toLane="0" via=":65_20_0" tl="65" linkIndex="20" dir="s" state="o"/>
    <connection from="65-1" to="66-1" fromLane="1" toLane="1" via=":65_20_1" tl="65" linkIndex="21" dir="s" state="o"/>
    <connection from="65-1" to="57-3" fromLane="1" toLane="3" via=":65_22_0" tl="65" linkIndex="22" dir="l" state="o"/>
    <connection from="65-1" to="64-5" fromLane="1" toLane="1" via=":65_23_0" tl="65" linkIndex="23" dir="t" state="o"/>
    <connection from="65-3" to="66-1" fromLane="0" toLane="0" via=":65_12_0" tl="65" linkIndex="12" dir="r" state="o"/>
    <connection from="65-3" to="57-3" fromLane="0" toLane="0" via=":65_13_0" tl="65" linkIndex="13" dir="s" state="o"/>
    <connection from="65-3" to="57-3" fromLane="1" toLane="1" via=":65_13_1" tl="65" linkIndex="14" dir="s" state="o"/>
    <connection from="65-3" to="57-3" fromLane="2" toLane="2" via=":65_13_2" tl="65" linkIndex="15" dir="s" state="o"/>
    <connection from="65-3" to="57-3" fromLane="3" toLane="3" via=":65_13_3" tl="65" linkIndex="16" dir="s" state="o"/>
    <connection from="65-3" to="64-5" fromLane="3" toLane="1" via=":65_17_0" tl="65" linkIndex="17" dir="l" state="o"/>
    <connection from="65-3" to="83-7" fromLane="3" toLane="3" via=":65_18_0" tl="65" linkIndex="18" dir="t" state="o"/>
    <connection from="65-5" to="57-3" fromLane="0" toLane="0" via=":65_7_0" tl="65" linkIndex="7" dir="r" state="o"/>
    <connection from="65-5" to="64-5" fromLane="0" toLane="0" via=":65_8_0" tl="65" linkIndex="8" dir="s" state="o"/>
    <connection from="65-5" to="64-5" fromLane="1" toLane="1" via=":65_8_1" tl="65" linkIndex="9" dir="s" state="o"/>
    <connection from="65-5" to="83-7" fromLane="1" toLane="3" via=":65_10_0" tl="65" linkIndex="10" dir="l" state="o"/>
    <connection from="65-5" to="66-1" fromLane="1" toLane="1" via=":65_11_0" tl="65" linkIndex="11" dir="t" state="o"/>
    <connection from="65-7" to="64-5" fromLane="0" toLane="0" via=":65_0_0" tl="65" linkIndex="0" dir="r" state="o"/>
    <connection from="65-7" to="83-7" fromLane="0" toLane="0" via=":65_1_0" tl="65" linkIndex="1" dir="s" state="o"/>
    <connection from="65-7" to="83-7" fromLane="1" toLane="1" via=":65_1_1" tl="65" linkIndex="2" dir="s" state="o"/>
    <connection from="65-7" to="83-7" fromLane="2" toLane="2" via=":65_1_2" tl="65" linkIndex="3" dir="s" state="o"/>
    <connection from="65-7" to="83-7" fromLane="3" toLane="3" via=":65_1_3" tl="65" linkIndex="4" dir="s" state="o"/>
    <connection from="65-7" to="66-1" fromLane="3" toLane="1" via=":65_5_0" tl="65" linkIndex="5" dir="l" state="o"/>
    <connection from="65-7" to="57-3" fromLane="3" toLane="3" via=":65_6_0" tl="65" linkIndex="6" dir="t" state="o"/>

    <connection from=":65_0" to="64-5" fromLane="0" toLane="0" dir="r" state="M"/>
    <connection from=":65_1" to="83-7" fromLane="0" toLane="0" dir="s" state="M"/>
    <connection from=":65_1" to="83-7" fromLane="1" toLane="1" dir="s" state="M"/>
    <connection from=":65_1" to="83-7" fromLane="2" toLane="2" dir="s" state="M"/>
    <connection from=":65_1" to="83-7" fromLane="3" toLane="3" dir="s" state="M"/>
    <connection from=":65_5" to="66-1" fromLane="0" toLane="1" via=":65_24_0" dir="l" state="m"/>
    <connection from=":65_24" to="66-1" fromLane="0" toLane="1" dir="l" state="M"/>
    <connection from=":65_6" to="57-3" fromLane="0" toLane="3" via=":65_25_0" dir="t" state="m"/>
    <connection from=":65_25" to="57-3" fromLane="0" toLane="3" dir="t" state="M"/>
    <connection from=":65_7" to="57-3" fromLane="0" toLane="0" dir="r" state="M"/>
    <connection from=":65_8" to="64-5" fromLane="0" toLane="0" dir="s" state="M"/>
    <connection from=":65_8" to="64-5" fromLane="1" toLane="1" dir="s" state="M"/>
    <connection from=":65_10" to="83-7" fromLane="0" toLane="3" via=":65_26_0" dir="l" state="m"/>
    <connection from=":65_26" to="83-7" fromLane="0" toLane="3" dir="l" state="M"/>
    <connection from=":65_11" to="66-1" fromLane="0" toLane="1" via=":65_27_0" dir="t" state="m"/>
    <connection from=":65_27" to="66-1" fromLane="0" toLane="1" dir="t" state="M"/>
    <connection from=":65_12" to="66-1" fromLane="0" toLane="0" dir="r" state="M"/>
    <connection from=":65_13" to="57-3" fromLane="0" toLane="0" dir="s" state="M"/>
    <connection from=":65_13" to="57-3" fromLane="1" toLane="1" dir="s" state="M"/>
    <connection from=":65_13" to="57-3" fromLane="2" toLane="2" dir="s" state="M"/>
    <connection from=":65_13" to="57-3" fromLane="3" toLane="3" dir="s" state="M"/>
    <connection from=":65_17" to="64-5" fromLane="0" toLane="1" via=":65_28_0" dir="l" state="m"/>
    <connection from=":65_28" to="64-5" fromLane="0" toLane="1" dir="l" state="M"/>
    <connection from=":65_18" to="83-7" fromLane="0" toLane="3" via=":65_29_0" dir="t" state="m"/>
    <connection from=":65_29" to="83-7" fromLane="0" toLane="3" dir="t" state="M"/>
    <connection from=":65_19" to="83-7" fromLane="0" toLane="0" dir="r" state="M"/>
    <connection from=":65_20" to="66-1" fromLane="0" toLane="0" dir="s" state="M"/>
    <connection from=":65_20" to="66-1" fromLane="1" toLane="1" dir="s" state="M"/>
    <connection from=":65_22" to="57-3" fromLane="0" toLane="3" via=":65_30_0" dir="l" state="m"/>
    <connection from=":65_30" to="57-3" fromLane="0" toLane="3" dir="l" state="M"/>
    <connection from=":65_23" to="64-5" fromLane="0" toLane="1" via=":65_31_0" dir="t" state="m"/>
    <connection from=":65_31" to="64-5" fromLane="0" toLane="1" dir="t" state="M"/>""", file=net)
        print("</net>", file=net)


def generate_cfgfile():
    with open("65.sumocfg", "w", encoding="utf-8") as cfg:
        print(
            u"""<?xml version="1.0" encoding="UTF-8"?>
        
        <configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
        
            <input>
                <net-file value="65.net.xml"/>
                <route-files value="65.rou.xml"/>
            </input>
            <output>
                <tripinfo-output value="tripinfo.output.xml"/>
            </output>
            <time>
                <begin value="0"/>
            </time>
            <processing>
                <ignore-route-errors value="true"/>
            </processing>
            <traci_server>
                <remote-port value="43536"/>
            </traci_server>
            <report>
            </report>
        
        </configuration>
            """, file=cfg)


def simulation(flow, signal):
    sumoBinary = checkBinary('sumo')

    generate_cfgfile()

    generate_netfile(signal)

    # first, generate the route file for this simulation
    generate_routefile(flow, signal)

    # this is the normal way of using traci. sumo is started as a
    # subprocess and then the python script connects and runs
    traci.start([sumoBinary, "-c", "65.sumocfg",
                 "--tripinfo-output", "tripinfo.xml"
                 ])
    run()
    return output_results()


# this is the main entry point of this script
def main():
    flow = {
        '65-1-57-3': 10,
        '65-1-66-1': 11,
        '65-5-83-7': 11,
        '65-5-64-5': 11,
        '65-3-57-3': 11,
        '65-7-83-7': 30
    }
    signal = {
        'cycle': 74,
        '65-1-57-3': 31,
        '65-5-83-7': 6,
        '65-5-64-5': 31,
        '65-3-57-3': 6
    }
    simulated_result = simulation(flow, signal)
    print(simulated_result)


if __name__ == "__main__":
    main()
