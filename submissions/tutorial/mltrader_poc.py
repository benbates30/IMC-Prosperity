import numpy as np
import pandas as pd
from numpy import array
from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order

log_reg_hardcoded_params = {'PEARLS': [{'learning_rate': 0.01, 'num_iter': 100, 'fit_intercept': True, 'verbose': False, 'weights': [-0.008281989704984687, -80.28541743615081, -79.8580066135332, -83.09641346370512, -81.90725673005745, -80.1083364548577, -76.71016781026272, -85.26647982077496, -84.06704977290083, -80.02889052934283, -79.98897336676627, -80.36176174821317, -79.93474937769776, -83.17697785918901, -81.98556518230374, -80.18546581846218, -76.78405793675817, -85.3482341653168, -84.14767537727592, -80.10594512090853, -80.0659562525895, -80.3257337500001, -79.9029527499999, -83.1399846250002, -81.94592603125012, -80.14418821874986, -76.74526435937514, -85.30598079687485, -84.10690150976554, -80.06723298144539, -80.02782921142582, -0.030429999999999947, -0.03641249999999997, -0.057047500000000105, -0.0924325, -0.19283500000000015, -0.16304250000000017, -0.11206000000000008, -0.22007750000000026, -0.07937750000000017, 0.3760274999999995, -80.29846499999985, -80.35300250000003, -79.83488749999985, -79.88101749999987, -83.89037250000001, -83.9259000000002, -82.20571000000012, -82.21049500000007, -79.20222499999998, -79.1868849999999, -72.41460500000008, -72.38235999999992, -93.81875999999993, -93.78123999999993, -89.81796000000007, -89.78203999999994, -69.61392000000009, -69.58607999999981, -69.21384, -69.18615999999986]}, {'learning_rate': 0.01, 'num_iter': 100, 'fit_intercept': True, 'verbose': False, 'weights': [-0.0022369024465642034, -19.865168546855486, -19.752881264341383, -19.55319021057325, -19.203012592707267, -18.603610986753502, -17.604320522463684, -16.004997324388736, -13.606012624807295, -15.40515705654653, -15.705130930961348, -19.88464642755646, -19.77190811753067, -19.57208325442945, -19.22200411881417, -18.621517124411625, -17.621057518946348, -16.020336742130326, -13.619063255331767, -15.420038611983431, -15.720113920669748, -19.8743187500001, -19.760686250000077, -19.562417499999967, -19.212229218750032, -18.61274999999985, -17.613096406250126, -16.01322636718738, -13.61215178710951, -15.41249562011734, -15.713165612793007, -0.0008500000000000001, 0.007787499999999982, 0.01718750000000002, 0.0319875, 0.08642500000000007, 0.1320124999999999, 0.13105, 0.20673750000000007, 0.4357375000000005, 0.8615124999999995, -19.865125000000106, -19.88351250000015, -19.736687500000222, -19.75796250000002, -19.487037500000113, -19.509050000000087, -18.99605000000006, -19.00712499999986, -18.002375000000118, -17.999074999999884, -16.00607500000024, -15.994699999999895, -12.002399999999895, -11.997600000000155, -4.000799999999915, -3.999200000000109, -13.002600000000044, -12.997399999999981, -16.003200000000128, -15.99679999999996]}, {'learning_rate': 0.01, 'num_iter': 100, 'fit_intercept': True, 'verbose': False, 'weights': [-0.00788190525249574, -76.2861284292824, -76.21979379952079, -76.09965565262931, -75.89056210535121, -75.29123630390698, -74.29208508020385, -72.69294783854154, -69.33443782996603, -66.85542611028549, -66.87501223237652, -76.3658560829688, -76.29543138177371, -76.17425113129654, -75.96396079568065, -75.36387255162444, -74.36357782684227, -72.76255189864607, -69.40094220646563, -66.91963857015124, -66.93984154953395, -76.32941625000007, -76.25828424999985, -76.13600674999984, -75.92844190625019, -75.32825143749987, -74.32837339062512, -72.72824518749988, -69.36845054101578, -66.88755815722635, -66.90711476025393, 0.0001100000000000002, -0.009832500000000006, -0.057477499999999966, -0.05119250000000009, -0.04477500000000009, 0.0763875, 0.2800699999999999, 0.2786324999999998, 0.2740324999999997, 0.38351750000000073, -76.28997000000015, -76.36886250000002, -76.2186675000001, -76.28214249999988, -76.06217250000022, -76.12268999999999, -75.78771000000013, -75.80970499999994, -74.80664500000016, -74.7912449999999, -72.810185, -72.78928000000015, -68.81375999999999, -68.78623999999986, -57.611520000000006, -57.58847999999986, -45.209039999999845, -45.19096000000005, -45.40907999999999, -45.39092000000011]}, {'learning_rate': 0.01, 'num_iter': 100, 'fit_intercept': True, 'verbose': False, 'weights': [-0.007911963575044518, -76.58913747061845, -76.25024027874979, -75.65157202961771, -74.6018792185621, -72.80258964062786, -71.90292337347904, -80.458998208873, -77.22041563177218, -77.18042980525243, -77.64003865772173, -76.65973427073638, -76.32395540925678, -75.72409104867903, -74.67346451111452, -72.87233154772706, -71.97192533549133, -80.53625822950731, -77.2949561800286, -77.25494218896198, -77.71481491777782, -76.61839874999994, -76.28399875000004, -75.69253000000012, -74.6390532812498, -72.83746328125002, -71.93620192187494, -80.49690777343774, -77.25805757617215, -77.21776404785172, -77.67625314892592, -0.03674000000000005, -0.0365425, 0.0064424999999999925, 0.053662500000000106, 0.03930499999999997, -0.025512500000000035, -0.08797500000000019, -0.1740774999999998, 0.06812250000000014, 0.014027499999999922, -76.58965000000016, -76.64714750000003, -76.22361749999996, -76.26651249999976, -75.50057249999983, -75.52086000000007, -74.02243000000006, -73.98938499999996, -71.01964499999993, -70.98231499999987, -69.21496499999986, -69.1799999999999, -90.61811999999992, -90.58187999999994, -79.81595999999995, -79.78403999999982, -79.61592000000022, -79.5840799999998, -84.21683999999985, -84.18315999999989]}, {'learning_rate': 0.01, 'num_iter': 100, 'fit_intercept': True, 'verbose': False, 'weights': [-0.0032368905465055744, -29.86228324908538, -29.749418454433943, -29.549246082572616, -29.19917513788552, -28.598929781093418, -27.599447970468393, -26.00025510715515, -28.099121165048487, -25.900145781960504, -26.19997652193422, -29.891943330296154, -29.777813963172264, -29.577647296417737, -29.22703102442169, -28.6265320008446, -27.62585485921587, -26.02506695291667, -28.1260104690352, -25.92489491436065, -26.225158353417324, -29.879318749999857, -29.764561249999808, -29.565167499999923, -29.21588546874986, -28.612765625000172, -27.61261203125003, -26.012827929687663, -28.112909990234193, -25.912534682617263, -26.212877624511634, 0.012649999999999982, 0.009787499999999998, 0.006187499999999989, 0.002987500000000002, -0.015574999999999943, -0.12648750000000025, -0.06945000000000014, 0.00023750000000008183, -0.1982625000000003, -0.2239875000000003, -29.870124999999792, -29.88851249999985, -29.748687499999846, -29.763962499999977, -29.50103749999989, -29.510050000000184, -29.00904999999988, -29.005125000000007, -28.00537500000012, -27.994075000000066, -26.005074999999895, -25.995699999999783, -22.004399999999976, -21.995599999999982, -29.00580000000005, -28.99420000000002, -18.00360000000009, -17.996399999999923, -21.004200000000157, -20.99579999999996]}, {'learning_rate': 0.01, 'num_iter': 100, 'fit_intercept': True, 'verbose': False, 'weights': [-0.009341973644481253, -90.88365882516972, -90.6342960892719, -90.19456469963691, -92.92356781601617, -94.84227917244375, -93.442363402226, -91.2032651755603, -93.84231785547959, -92.36317607720065, -90.38426325027635, -90.96831153537886, -90.72137193836257, -90.28056206139108, -93.01178886107357, -94.93274327288415, -93.5318765163712, -91.29104208427732, -93.93256193088125, -92.45191671163283, -90.47092109479786, -90.92901374999991, -90.68174975000004, -90.23883974999981, -92.96956959375004, -94.88744215625006, -93.48696306250012, -91.24637078906238, -93.88724489257811, -92.40777477441407, -90.42752786181619, 0.01777999999999996, 0.034077500000000024, 0.016887499999999993, 0.03302249999999999, -0.06500500000000006, -0.24638249999999973, -0.27442999999999973, -0.2928674999999995, -0.48896749999999917, -0.1874824999999996, -90.89865000000002, -90.95937750000009, -90.62986249999994, -90.6826625000001, -90.08400249999984, -90.12461999999995, -94.00310999999986, -94.01078500000004, -97.21112500000008, -97.19410500000016, -94.41950499999999, -94.38697999999995, -88.8177599999999, -88.78224000000009, -97.6195199999999, -97.58048000000018, -90.21803999999997, -90.18196000000017, -70.4140800000001, -70.38592000000008]}, {'learning_rate': 0.01, 'num_iter': 100, 'fit_intercept': True, 'verbose': False, 'weights': [0.0033421185023631934, 30.91102793796611, 29.941628862770923, 32.221296979747535, 32.711637119736764, 30.55251927906401, 24.455585882995802, 38.689228667253275, 37.54962613659198, 32.31196320182529, 34.5706026632124, 30.942902257395346, 29.973157430575057, 32.2538219162449, 32.742471479143134, 30.58177090045996, 24.4789001690011, 38.72608662877248, 37.585865818778075, 32.34356728720259, 34.60463782747921, 30.931043750000185, 29.96460175000008, 32.247351124999724, 32.733517218750094, 30.565306406249913, 24.464952531249963, 38.706085624999886, 37.56831984179689, 32.327855249023486, 34.58728196240246, 0.03783999999999998, 0.03181250000000002, 0.06917750000000011, 0.02431749999999994, 0.19623000000000038, 0.3575575000000006, 0.10081499999999975, 0.10547250000000025, -0.21470749999999994, -1.3810975000000005, 30.90356500000001, 30.958522500000115, 29.827662499999985, 29.89829750000007, 32.67187250000015, 32.76492500000006, 33.42278000000023, 33.42669499999997, 29.80679500000001, 29.781295000000153, 17.608145000000093, 17.589230000000097, 53.210639999999934, 53.189359999999944, 49.409880000000236, 49.39012000000011, 23.204640000000083, 23.195359999999763, 45.809160000000034, 45.79083999999996]}], 'BANANAS': [{'learning_rate': 0.01, 'num_iter': 100, 'fit_intercept': True, 'verbose': False, 'weights': [-0.0005369696119910463, -1.4467134314092611, -1.3028447474069615, -1.0454011790344646, -0.595706216978912, -1.3081767741593429, -3.7325477317396207, -6.619117845975992, -4.010910325651173, -4.488963860720727, -4.21210146132927, -1.453617481654177, -1.3058708790849534, -1.046339443863599, -0.595618346857786, -1.3098432796740314, -3.737461043806629, -6.628045717275363, -4.016508725754562, -4.495048943287959, -4.217673962141518, -1.4482687500000067, -1.3022205000000442, -1.0464306875000249, -0.596564624999945, -1.3074964687500166, -3.7338934765624447, -6.622804722656226, -4.013315160156127, -4.491750059570405, -4.215558781738153, -0.02549499999999995, -0.02995749999999993, -0.03390749999999994, -0.03682249999999993, -0.21300750000000027, -0.17374500000000018, -0.3332850000000005, -0.47930250000000113, -0.5792474999999987, -0.6857274999999994, -1.4404625000000384, -1.456074999999966, -1.2816574999999055, -1.2826899999999277, -0.9590924999999944, -0.9599100000000247, -0.31826999999996985, -0.31765750000002413, -1.505069999999979, -1.5063999999999451, -6.3675174999999635, -6.3403500000001, -13.599052500000125, -13.573902500000116, -4.886867500000063, -4.867472499999959, -7.267070000000061, -7.2418625, -4.568472500000054, -4.568729999999952]}, {'learning_rate': 0.01, 'num_iter': 100, 'fit_intercept': True, 'verbose': False, 'weights': [-0.004226929238466444, -19.64120212301474, -19.539834917006836, -19.361960728458236, -19.050919071942335, -22.970752410583593, -22.077016453568703, -20.650357149113425, -19.254367095256057, -19.872977062702095, -20.504270563326145, -19.666799629885848, -19.56592193568479, -19.388286116838458, -19.077360996196273, -23.002161629299405, -22.106779536841152, -20.6782887778948, -19.280533607686564, -19.899942758255218, -20.532402901353723, -19.653568749999884, -19.549020249999877, -19.372208937500076, -19.060868875000033, -22.98447846874998, -22.090753476562433, -20.663469050781238, -19.267570488281123, -19.88645681933602, -20.5178097475585, 0.002899999999999997, -0.021222500000000026, -0.03265250000000002, -0.11419249999999999, -0.04194750000000006, -0.015404999999999988, -0.04791500000000005, -0.04074750000000013, -0.2922025000000008, -1.1343224999999981, -19.650127499999936, -19.65700999999997, -19.535282499999976, -19.5338849999999, -19.32339750000007, -19.302784999999865, -18.88540999999995, -18.85854249999985, -25.426794999999927, -25.388089999999927, -23.63577749999996, -23.589099999999917, -20.068997500000034, -20.04064749999997, -15.41448249999993, -15.390477500000006, -18.506829999999958, -18.452387499999965, -24.675477499999978, -24.57987000000002]}, {'learning_rate': 0.01, 'num_iter': 100, 'fit_intercept': True, 'verbose': False, 'weights': [-0.00021717958698141873, 0.030182080389018928, -1.603412374775476, -2.530901549010001, -2.421828245539901, -5.198218072947543, -1.1787247825821145, 5.260858326791638, 9.720884524316165, -11.805257791446842, -9.8947013881282, 0.0344556935438618, -1.6025795363522048, -2.5330545133537, -2.4247368700267478, -5.204429911206021, -1.1800441668323636, 5.268375394827298, 9.73420851190113, -11.821617418421138, -9.908380605954935, 0.037063750000045026, -1.5960192500000199, -2.5280306874998733, -2.4158827499999838, -5.194587718749953, -1.1733467578124577, 5.264458519531194, 9.727350039062404, -11.811396151367106, -9.89551534423828, 0.009519999999999989, -0.022132499999999996, 0.022732499999999965, -0.12180249999999994, -0.16201749999999998, -0.607235000000001, -0.06047000000000007, -1.1119874999999986, -2.665097500000001, -4.110967500000001, 0.03255249999999332, 0.04157499999999015, -1.779452499999996, -1.762775000000043, -2.944387500000053, -2.9243549999999665, -2.788384999999984, -2.7390325000000706, -7.419694999999873, -7.375645000000053, 0.634432500000055, 0.6214600000000328, 16.718337499999997, 16.65326250000003, 31.509597500000027, 31.384162500000034, -76.28211999999998, -76.17412249999991, -56.72143249999996, -56.55161000000004]}, {'learning_rate': 0.01, 'num_iter': 100, 'fit_intercept': True, 'verbose': False, 'weights': [-0.004061949168795606, -18.81345945864352, -18.68897343045625, -18.469841512282244, -18.019128220931695, -17.131742587727746, -18.12421415433071, -19.71249520793951, -19.60569510827022, -17.618417861868714, -17.558021334588215, -18.837908433332245, -18.714794436186608, -18.495475448228305, -18.043949534893358, -17.156085275001807, -18.14891505400729, -19.739330145456705, -19.632550460288385, -17.642486136527573, -17.581688562547104, -18.829233750000043, -18.702902250000044, -18.479213562499936, -18.02813937499999, -17.14398437500004, -18.136621054687538, -19.725621074218907, -19.618842855468756, -17.630394317382688, -17.570546561035147, 0.026524999999999976, 0.05303749999999995, 0.06339750000000009, 0.0058775000000000025, 0.055927500000000074, 0.036415000000000024, -0.18590499999999965, -0.2258025000000003, -0.08183750000000001, 0.1724624999999996, -18.81744249999994, -18.841025000000002, -18.67919750000005, -18.695830000000015, -18.40073249999996, -18.410390000000042, -17.758534999999924, -17.75969749999999, -16.289795000000034, -16.28167499999999, -18.2945874999999, -18.25100000000001, -22.258277500000005, -22.222432500000068, -21.87462750000006, -21.830732500000032, -11.882550000000066, -11.843812500000102, -11.36746250000009, -11.341949999999994]}, {'learning_rate': 0.01, 'num_iter': 100, 'fit_intercept': True, 'verbose': False, 'weights': [-0.00661695190003239, -31.3900798664036, -31.33484654494626, -31.238248821861212, -31.06906218208022, -30.77662660761387, -30.281324940859697, -29.48281351235569, -29.780722862915766, -29.190539913479228, -27.31599644012607, -31.432065065857707, -31.37630965552231, -31.280327152449413, -31.11118997741089, -30.81817587720924, -30.322162442342528, -29.52240132091984, -29.82087579525858, -29.229710029652118, -27.35302083963679, -31.411521250000064, -31.35641124999998, -31.259997187500062, -31.092077499999988, -30.79996859374997, -30.303467851562445, -29.50383615234368, -29.800875234375006, -29.210287290039126, -27.335168431152226, -0.008454999999999985, -0.055902499999999924, -0.05068250000000004, -0.0012075000000000096, -0.03629249999999995, -0.023204999999999948, -0.056569999999999926, -0.3026925000000004, 0.1861674999999999, 1.9602725000000027, -31.39232249999995, -31.43072000000009, -31.338477500000035, -31.363804999999953, -31.22384749999996, -31.238135000000085, -30.988865000000008, -30.990792499999902, -30.511834999999884, -30.483210000000007, -29.538142499999893, -29.49255500000003, -27.538757500000017, -27.505492499999946, -28.532747499999992, -28.488472499999986, -25.55621500000008, -25.51695250000003, -6.921472500000027, -6.919229999999967]}, {'learning_rate': 0.01, 'num_iter': 100, 'fit_intercept': True, 'verbose': False, 'weights': [-0.0029770578975513583, -12.928325286854452, -12.562854367399389, -11.87383244823993, -10.667769693461395, -8.479021144131835, -4.638508131635119, -10.090143442261537, -15.27557044043973, -6.057790943286754, -9.44785563864714, -12.941541215783234, -12.57795378133879, -11.888028831527238, -10.680321895735325, -8.488825324702312, -4.644289413242207, -10.102645488773236, -15.294436139625908, -6.063921691783806, -9.458424247188274, -12.940451249999995, -12.578938000000143, -11.889084562500114, -10.68506143750008, -8.492516156250046, -4.646682492187397, -10.098756324218824, -15.287581964843762, -6.060281983398509, -9.456306636230407, 0.023569999999999997, 0.07846749999999993, 0.07906750000000008, 0.14183249999999986, 0.05190750000000013, -0.12218500000000009, -0.9924149999999983, -1.4584075000000025, 0.8920674999999987, 6.199577499999995, -12.933227500000042, -12.947674999999935, -12.537992500000009, -12.54659499999998, -11.670972500000174, -11.683979999999984, -9.963155000000093, -9.967412500000059, -6.340660000000026, -6.320935000000043, 1.3924725000000358, 1.3442400000001733, -12.21644249999993, -12.195417499999927, -29.475362499999996, -29.424997499999925, 16.81516000000011, 16.722792500000004, -18.084877499999926, -18.264070000000025]}, {'learning_rate': 0.01, 'num_iter': 100, 'fit_intercept': True, 'verbose': False, 'weights': [-0.003947322664434631, -21.185983080261604, -20.42084747030793, -20.993005712054376, -23.723842548154256, -19.625285106719524, -24.96504565972272, -21.96682914515116, -23.396617009424634, -12.888656104279267, -8.894887350715297, -21.221981336791853, -20.45395958076531, -21.025370079571196, -23.758703554056257, -19.653763254051785, -25.000474340596003, -21.998867550899913, -23.430384569936486, -12.908579914360207, -8.909540422208789, -21.200678750000062, -20.439209000000112, -21.009335937499838, -23.740752250000074, -19.637783406249987, -24.98332972656252, -21.98026686328108, -23.410291386718658, -12.900552366210768, -8.902861644042869, -0.03451999999999997, -0.01412750000000002, -0.07153750000000005, 0.0808825, 0.25676750000000054, 0.7244349999999989, 1.298694999999998, 2.9519924999999967, 1.6235075000000008, -2.6914025, -21.189242499999928, -21.21211500000009, -20.338052500000227, -20.389840000000024, -21.061332499999963, -21.10923000000012, -24.955670000000005, -25.022032500000158, -18.092764999999975, -18.176210000000008, -28.81293750000009, -28.814710000000016, -21.319887499999894, -21.271007500000106, -26.079102499999898, -25.982797500000096, 26.31773999999996, 26.349332500000056, 67.16020249999995, 67.20867000000018]}]}
dec_tree_hardcoded_params = {'PEARLS': [{'max_depth': 3, 'min_samples_split': 2, 'min_samples_leaf': 1, 'classes': array([0, 1]), 'Tree': {'column': 17, 'threshold': 10004.859290438966, 'probas': array([0.9515, 0.0485]), 'is_terminal': False, 'depth': 1, 'left': {'column': 27, 'threshold': 10000.28515625, 'probas': array([0.95197599, 0.04802401]), 'is_terminal': False, 'depth': 2, 'left': {'column': None, 'threshold': None, 'probas': array([0.95245245, 0.04754755]), 'is_terminal': True, 'depth': 3}, 'right': {'column': None, 'threshold': None, 'probas': array([0., 1.]), 'is_terminal': True, 'depth': 3}}, 'right': {'column': None, 'threshold': None, 'probas': array([0., 1.]), 'is_terminal': True, 'depth': 2}}}, {'max_depth': 3, 'min_samples_split': 2, 'min_samples_leaf': 1, 'classes': array([0, 1]), 'Tree': {'column': 8, 'threshold': 9995.249345740041, 'probas': array([0.977, 0.023]), 'is_terminal': False, 'depth': 1, 'left': {'column': 38, 'threshold': -175.0, 'probas': array([0.97747748, 0.02252252]), 'is_terminal': False, 'depth': 2, 'left': {'column': None, 'threshold': None, 'probas': array([0.8, 0.2]), 'is_terminal': True, 'depth': 3}, 'right': {'column': None, 'threshold': None, 'probas': array([0.97837022, 0.02162978]), 'is_terminal': True, 'depth': 3}}, 'right': {'column': 0, 'threshold': 9995.0, 'probas': array([0.5, 0.5]), 'is_terminal': False, 'depth': 2, 'left': {'column': None, 'threshold': None, 'probas': array([1., 0.]), 'is_terminal': True, 'depth': 3}, 'right': {'column': None, 'threshold': None, 'probas': array([0., 1.]), 'is_terminal': True, 'depth': 3}}}}, {'max_depth': 3, 'min_samples_split': 2, 'min_samples_leaf': 1, 'classes': array([0, 1]), 'Tree': {'column': 37, 'threshold': -145.0, 'probas': array([0.973, 0.027]), 'is_terminal': False, 'depth': 1, 'left': {'column': None, 'threshold': None, 'probas': array([0., 1.]), 'is_terminal': True, 'depth': 2}, 'right': {'column': 34, 'threshold': 60.0, 'probas': array([0.97348674, 0.02651326]), 'is_terminal': False, 'depth': 2, 'left': {'column': None, 'threshold': None, 'probas': array([0.97396094, 0.02603906]), 'is_terminal': True, 'depth': 3}, 'right': {'column': None, 'threshold': None, 'probas': array([0.5, 0.5]), 'is_terminal': True, 'depth': 3}}}}, {'max_depth': 3, 'min_samples_split': 2, 'min_samples_leaf': 1, 'classes': array([0, 1]), 'Tree': {'column': 15, 'threshold': 10004.92857142857, 'probas': array([0.9515, 0.0485]), 'is_terminal': False, 'depth': 1, 'left': {'column': 33, 'threshold': 44.0, 'probas': array([0.95197599, 0.04802401]), 'is_terminal': False, 'depth': 2, 'left': {'column': None, 'threshold': None, 'probas': array([0.95245245, 0.04754755]), 'is_terminal': True, 'depth': 3}, 'right': {'column': None, 'threshold': None, 'probas': array([0., 1.]), 'is_terminal': True, 'depth': 3}}, 'right': {'column': None, 'threshold': None, 'probas': array([0., 1.]), 'is_terminal': True, 'depth': 2}}}, {'max_depth': 3, 'min_samples_split': 2, 'min_samples_leaf': 1, 'classes': array([0, 1]), 'Tree': {'column': 21, 'threshold': 10003.25, 'probas': array([0.978, 0.022]), 'is_terminal': False, 'depth': 1, 'left': {'column': 23, 'threshold': 9998.875, 'probas': array([0.97847848, 0.02152152]), 'is_terminal': False, 'depth': 2, 'left': {'column': None, 'threshold': None, 'probas': array([0.89361702, 0.10638298]), 'is_terminal': True, 'depth': 3}, 'right': {'column': None, 'threshold': None, 'probas': array([0.98052281, 0.01947719]), 'is_terminal': True, 'depth': 3}}, 'right': {'column': 0, 'threshold': 9995.225806451614, 'probas': array([0.5, 0.5]), 'is_terminal': False, 'depth': 2, 'left': {'column': None, 'threshold': None, 'probas': array([0., 1.]), 'is_terminal': True, 'depth': 3}, 'right': {'column': None, 'threshold': None, 'probas': array([1., 0.]), 'is_terminal': True, 'depth': 3}}}}, {'max_depth': 3, 'min_samples_split': 2, 'min_samples_leaf': 1, 'classes': array([0, 1]), 'Tree': {'column': 15, 'threshold': 10004.915151515152, 'probas': array([0.974, 0.026]), 'is_terminal': False, 'depth': 1, 'left': {'column': 31, 'threshold': 26.0, 'probas': array([0.97574533, 0.02425467]), 'is_terminal': False, 'depth': 2, 'left': {'column': None, 'threshold': None, 'probas': array([0.97622661, 0.02377339]), 'is_terminal': True, 'depth': 3}, 'right': {'column': None, 'threshold': None, 'probas': array([0.5, 0.5]), 'is_terminal': True, 'depth': 3}}, 'right': {'column': 11, 'threshold': 10004.98076923077, 'probas': array([0.80952381, 0.19047619]), 'is_terminal': False, 'depth': 2, 'left': {'column': None, 'threshold': None, 'probas': array([0.89473684, 0.10526316]), 'is_terminal': True, 'depth': 3}, 'right': {'column': None, 'threshold': None, 'probas': array([0., 1.]), 'is_terminal': True, 'depth': 3}}}}, {'max_depth': 3, 'min_samples_split': 2, 'min_samples_leaf': 1, 'classes': array([0, 1]), 'Tree': {'column': 22, 'threshold': 9999.125, 'probas': array([0.195, 0.805]), 'is_terminal': False, 'depth': 1, 'left': {'column': 6, 'threshold': 9995.274643874644, 'probas': array([0.26153846, 0.73846154]), 'is_terminal': False, 'depth': 2, 'left': {'column': None, 'threshold': None, 'probas': array([0.24505929, 0.75494071]), 'is_terminal': True, 'depth': 3}, 'right': {'column': None, 'threshold': None, 'probas': array([0.85714286, 0.14285714]), 'is_terminal': True, 'depth': 3}}, 'right': {'column': 6, 'threshold': 9995.237190558433, 'probas': array([0.18505747, 0.81494253]), 'is_terminal': False, 'depth': 2, 'left': {'column': None, 'threshold': None, 'probas': array([0.1984184, 0.8015816]), 'is_terminal': True, 'depth': 3}, 'right': {'column': None, 'threshold': None, 'probas': array([0.13180516, 0.86819484]), 'is_terminal': True, 'depth': 3}}}}], 'BANANAS': [{'max_depth': 3, 'min_samples_split': 2, 'min_samples_leaf': 1, 'classes': array([0, 1]), 'Tree': {'column': 36, 'threshold': 85.0, 'probas': array([0.945, 0.055]), 'is_terminal': False, 'depth': 1, 'left': {'column': 52, 'threshold': 4931.0, 'probas': array([0.94594595, 0.05405405]), 'is_terminal': False, 'depth': 2, 'left': {'column': None, 'threshold': None, 'probas': array([0.83695652, 0.16304348]), 'is_terminal': True, 'depth': 3}, 'right': {'column': None, 'threshold': None, 'probas': array([0.95120672, 0.04879328]), 'is_terminal': True, 'depth': 3}}, 'right': {'column': None, 'threshold': None, 'probas': array([0., 1.]), 'is_terminal': True, 'depth': 2}}}, {'max_depth': 3, 'min_samples_split': 2, 'min_samples_leaf': 1, 'classes': array([0, 1]), 'Tree': {'column': 33, 'threshold': -30.0, 'probas': array([0.959, 0.041]), 'is_terminal': False, 'depth': 1, 'left': {'column': 6, 'threshold': 4927.189895470383, 'probas': array([0.25, 0.75]), 'is_terminal': False, 'depth': 2, 'left': {'column': None, 'threshold': None, 'probas': array([1., 0.]), 'is_terminal': True, 'depth': 3}, 'right': {'column': None, 'threshold': None, 'probas': array([0., 1.]), 'is_terminal': True, 'depth': 3}}, 'right': {'column': 9, 'threshold': 4945.953479933231, 'probas': array([0.96042084, 0.03957916]), 'is_terminal': False, 'depth': 2, 'left': {'column': None, 'threshold': None, 'probas': array([0.96258847, 0.03741153]), 'is_terminal': True, 'depth': 3}, 'right': {'column': None, 'threshold': None, 'probas': array([0.72222222, 0.27777778]), 'is_terminal': True, 'depth': 3}}}}, {'max_depth': 3, 'min_samples_split': 2, 'min_samples_leaf': 1, 'classes': array([0, 1]), 'Tree': {'column': 57, 'threshold': 4925.0, 'probas': array([0.8395, 0.1605]), 'is_terminal': False, 'depth': 1, 'left': {'column': 27, 'threshold': 4947.8046875, 'probas': array([0.63476562, 0.36523438]), 'is_terminal': False, 'depth': 2, 'left': {'column': None, 'threshold': None, 'probas': array([0.72636816, 0.27363184]), 'is_terminal': True, 'depth': 3}, 'right': {'column': None, 'threshold': None, 'probas': array([0.3, 0.7]), 'is_terminal': True, 'depth': 3}}, 'right': {'column': 19, 'threshold': 4456.784879893033, 'probas': array([0.90994624, 0.09005376]), 'is_terminal': False, 'depth': 2, 'left': {'column': None, 'threshold': None, 'probas': array([0.48333333, 0.51666667]), 'is_terminal': True, 'depth': 3}, 'right': {'column': None, 'threshold': None, 'probas': array([0.92787115, 0.07212885]), 'is_terminal': True, 'depth': 3}}}}, {'max_depth': 3, 'min_samples_split': 2, 'min_samples_leaf': 1, 'classes': array([0, 1]), 'Tree': {'column': 30, 'threshold': 13.0, 'probas': array([0.9305, 0.0695]), 'is_terminal': False, 'depth': 1, 'left': {'column': 16, 'threshold': 4956.32618510158, 'probas': array([0.93176116, 0.06823884]), 'is_terminal': False, 'depth': 2, 'left': {'column': None, 'threshold': None, 'probas': array([0.93222892, 0.06777108]), 'is_terminal': True, 'depth': 3}, 'right': {'column': None, 'threshold': None, 'probas': array([0., 1.]), 'is_terminal': True, 'depth': 3}}, 'right': {'column': 30, 'threshold': 15.0, 'probas': array([0.57142857, 0.42857143]), 'is_terminal': False, 'depth': 2, 'left': {'column': None, 'threshold': None, 'probas': array([0., 1.]), 'is_terminal': True, 'depth': 3}, 'right': {'column': None, 'threshold': None, 'probas': array([1., 0.]), 'is_terminal': True, 'depth': 3}}}}, {'max_depth': 3, 'min_samples_split': 2, 'min_samples_leaf': 1, 'classes': array([0, 1]), 'Tree': {'column': 39, 'threshold': 232.0, 'probas': array([0.9565, 0.0435]), 'is_terminal': False, 'depth': 1, 'left': {'column': 21, 'threshold': 4955.0, 'probas': array([0.96452328, 0.03547672]), 'is_terminal': False, 'depth': 2, 'left': {'column': None, 'threshold': None, 'probas': array([0.96505824, 0.03494176]), 'is_terminal': True, 'depth': 3}, 'right': {'column': None, 'threshold': None, 'probas': array([0., 1.]), 'is_terminal': True, 'depth': 3}}, 'right': {'column': 46, 'threshold': 4944.0, 'probas': array([0.88265306, 0.11734694]), 'is_terminal': False, 'depth': 2, 'left': {'column': None, 'threshold': None, 'probas': array([0.9197861, 0.0802139]), 'is_terminal': True, 'depth': 3}, 'right': {'column': None, 'threshold': None, 'probas': array([0.11111111, 0.88888889]), 'is_terminal': True, 'depth': 3}}}}, {'max_depth': 3, 'min_samples_split': 2, 'min_samples_leaf': 1, 'classes': array([0, 1]), 'Tree': {'column': 15, 'threshold': 4935.1658823529415, 'probas': array([0.7605, 0.2395]), 'is_terminal': False, 'depth': 1, 'left': {'column': 8, 'threshold': 4930.033496967947, 'probas': array([0.94954955, 0.05045045]), 'is_terminal': False, 'depth': 2, 'left': {'column': None, 'threshold': None, 'probas': array([0.97490347, 0.02509653]), 'is_terminal': True, 'depth': 3}, 'right': {'column': None, 'threshold': None, 'probas': array([0.59459459, 0.40540541]), 'is_terminal': True, 'depth': 3}}, 'right': {'column': 29, 'threshold': 4455.3755859375, 'probas': array([0.68788927, 0.31211073]), 'is_terminal': False, 'depth': 2, 'left': {'column': None, 'threshold': None, 'probas': array([0.84660767, 0.15339233]), 'is_terminal': True, 'depth': 3}, 'right': {'column': None, 'threshold': None, 'probas': array([0.63924051, 0.36075949]), 'is_terminal': True, 'depth': 3}}}}, {'max_depth': 3, 'min_samples_split': 2, 'min_samples_leaf': 1, 'classes': array([0, 1]), 'Tree': {'column': 17, 'threshold': 4935.9699583581205, 'probas': array([0.609, 0.391]), 'is_terminal': False, 'depth': 1, 'left': {'column': 9, 'threshold': 4928.902495479204, 'probas': array([0.49455865, 0.50544135]), 'is_terminal': False, 'depth': 2, 'left': {'column': None, 'threshold': None, 'probas': array([0.38693467, 0.61306533]), 'is_terminal': True, 'depth': 3}, 'right': {'column': None, 'threshold': None, 'probas': array([0.77391304, 0.22608696]), 'is_terminal': True, 'depth': 3}}, 'right': {'column': 35, 'threshold': 32.0, 'probas': array([0.68968457, 0.31031543]), 'is_terminal': False, 'depth': 2, 'left': {'column': None, 'threshold': None, 'probas': array([0.72533849, 0.27466151]), 'is_terminal': True, 'depth': 3}, 'right': {'column': None, 'threshold': None, 'probas': array([0.42446043, 0.57553957]), 'is_terminal': True, 'depth': 3}}}}]}


def serialize_node(model):
    serialized_model = {
        'column':model.column,
        'threshold':model.threshold,
        'probas':model.probas,
        'is_terminal':model.is_terminal,
        'depth':model.depth
    }

    if not model.is_terminal:
        serialized_model['left'] = serialize_node(model.left)
        serialized_model['right'] = serialize_node(model.right)

    return serialized_model

def deserialize_node(model_dict):
    deserialized_node = Node()
    deserialized_node.column = model_dict['column']
    deserialized_node.threshold = model_dict['threshold']
    deserialized_node.probas = model_dict['probas']
    deserialized_node.is_terminal = model_dict['is_terminal']
    deserialized_node.depth = model_dict['depth']
    if not deserialized_node.is_terminal:
        deserialized_node.left = deserialize_node(model_dict['left'])
        deserialized_node.right = deserialize_node(model_dict['right'])
    return deserialized_node

def serialize_decision_tree(model):
    serialized_model = {
        'max_depth':model.max_depth,
        'min_samples_split':model.min_samples_split,
        'min_samples_leaf':model.min_samples_leaf,
        'classes':model.classes,
        'Tree': serialize_node(model.Tree)
    }
    return serialized_model

def deserialize_decision_tree(model_dict):
    deserialized_decision_tree = DecisionTreeClassifier()
    deserialized_decision_tree.max_depth = model_dict['max_depth']
    deserialized_decision_tree.min_samples_split = model_dict['min_samples_split']
    deserialized_decision_tree.min_samples_leaf = model_dict['min_samples_leaf']
    deserialized_decision_tree.classes = model_dict['classes']
    deserialized_decision_tree.Tree = deserialize_node(model_dict['Tree'])
    return deserialized_decision_tree

class Node:
    def __init__(self):
        
        # links to the left and right child nodes
        self.right = None
        self.left = None
        
        # derived from splitting criteria
        self.column = None
        self.threshold = None
        
        # probability for object inside the Node to belong for each of the given classes
        self.probas = None
        # depth of the given node
        self.depth = None
        
        # if it is the root Node or not
        self.is_terminal = False

class DecisionTreeClassifier:
    def __init__(self, max_depth = 3, min_samples_leaf = 1, min_samples_split = 2):
        
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        
        self.classes = None
        
        # Decision tree itself
        self.Tree = None
    
    def nodeProbas(self, y):
        '''
        Calculates probability of class in a given node
        '''
        
        probas = []
        
        # for each unique label calculate the probability for it
        for one_class in self.classes:
            proba = y[y == one_class].shape[0] / y.shape[0]
            probas.append(proba)
        return np.asarray(probas)

    def gini(self, probas):
        '''
        Calculates gini criterion
        '''
        
        return 1 - np.sum(probas**2)
    
    def calcImpurity(self, y):
        '''
        Wrapper for the impurity calculation. Calculates probas first and then passses them
        to the Gini criterion
        '''
        
        return self.gini(self.nodeProbas(y))
    
    def calcBestSplit(self, X, y):
        '''
        Calculates the best possible split for the concrete node of the tree
        '''
        
        bestSplitCol = None
        bestThresh = None
        bestInfoGain = -999
        
        impurityBefore = self.calcImpurity(y)
        
        # for each column in X
        for col in range(X.shape[1]):
            x_col = X[:, col]
            
            # for each value in the column
            for x_i in x_col:
                threshold = x_i
                y_right = y[x_col > threshold]
                y_left = y[x_col <= threshold]
                
                if y_right.shape[0] == 0 or y_left.shape[0] == 0:
                    continue
                    
                # calculate impurity for the right and left nodes
                impurityRight = self.calcImpurity(y_right)
                impurityLeft = self.calcImpurity(y_left)
                
                # calculate information gain
                infoGain = impurityBefore
                infoGain -= (impurityLeft * y_left.shape[0] / y.shape[0]) + (impurityRight * y_right.shape[0] / y.shape[0])
                
                # is this infoGain better then all other?
                if infoGain > bestInfoGain:
                    bestSplitCol = col
                    bestThresh = threshold
                    bestInfoGain = infoGain
                    
        
        # if we still didn't find the split
        if bestInfoGain == -999:
            return None, None, None, None, None, None
        
        # making the best split
        
        x_col = X[:, bestSplitCol]
        x_left, x_right = X[x_col <= bestThresh, :], X[x_col > bestThresh, :]
        y_left, y_right = y[x_col <= bestThresh], y[x_col > bestThresh]
        
        return bestSplitCol, bestThresh, x_left, y_left, x_right, y_right
                
    
    def buildDT(self, X, y, node):
        '''
        Recursively builds decision tree from the top to bottom
        '''
        
        # checking for the terminal conditions
        
        if node.depth >= self.max_depth:
            node.is_terminal = True
            return
        
        if X.shape[0] < self.min_samples_split:
            node.is_terminal = True
            return
        
        if np.unique(y).shape[0] == 1:
            node.is_terminal = True
            return
        
        # calculating current split
        splitCol, thresh, x_left, y_left, x_right, y_right = self.calcBestSplit(X, y)
        
        if splitCol is None:
            node.is_terminal = True
            
        if x_left.shape[0] < self.min_samples_leaf or x_right.shape[0] < self.min_samples_leaf:
            node.is_terminal = True
            return
        
        node.column = splitCol
        node.threshold = thresh
        
        # creating left and right child nodes
        node.left = Node()
        node.left.depth = node.depth + 1
        node.left.probas = self.nodeProbas(y_left)
        
        node.right = Node()
        node.right.depth = node.depth + 1
        node.right.probas = self.nodeProbas(y_right)
        
        # splitting recursevely
        self.buildDT(x_right, y_right, node.right)
        self.buildDT(x_left, y_left, node.left)
    
    def fit(self, X, y):
        '''
        Standard fit function to run all the model training
        '''
        if type(X) == pd.DataFrame:
            X = np.asarray(X)
        
        self.classes = np.unique(y)
        # root node creation
        self.Tree = Node()
        self.Tree.depth = 1
        self.Tree.probas = self.nodeProbas(y)
        
        self.buildDT(X, y, self.Tree)
    
    def predictSample(self, x, node):
        '''
        Passes one object through decision tree and return the probability of it to belong to each class
        '''
        # if we have reached the terminal node of the tree
        if node.is_terminal:
            return node.probas
        
        if x[node.column] > node.threshold:
            probas = self.predictSample(x, node.right)
        else:
            probas = self.predictSample(x, node.left)
            
        return probas
        
    def predict(self, X):
        '''
        Returns the labels for each X
        '''
        
        if type(X) == pd.DataFrame:
            X = np.asarray(X)
            
        predictions = []
        for x in X:
            pred = np.argmax(self.predictSample(x, self.Tree))
            predictions.append(pred)
        
        return np.asarray(predictions)
    
    def eval(self, X, y):
        """"Evaluate accuracy on dataset."""
        p = self.predict(X)
        return np.sum(p == y) / X.shape[0]
    
class Forest:
    def __init__(self, max_depth=5, no_trees=7,
                 min_samples_split=2, min_samples_leaf=1, feature_search=None,
                 bootstrap=True):
        """Random Forest implementation using numpy.
        Args:
            max_depth(int): Max depth of trees.
            no_trees(int): Number of trees.
            min_samples_split(int): Number of samples in a node to allow
            split search.
            min_samples_leaf(int): Number of samples to be deemed a leaf node.
            feature_search(int): Number of features to search when splitting.
            bootstrap(boolean): Resample dataset with replacement
        """
        self._trees = []
        self._max_depth = max_depth
        self._no_trees = no_trees
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf
        self._feature_search = feature_search
        self._bootstrap = bootstrap

    def train(self, x, y):
        """Training procedure.
        Args:
            x(ndarray): Inputs.
            y(ndarray): Labels.
        Returns:
            None
        """
        print('Training Forest...\n')
        for i in range(self._no_trees):
            print('\nTraining Decision Tree no {}...\n'.format(i + 1))
            tree = DecisionTreeClassifier(max_depth=self._max_depth,
                        min_samples_split=self._min_samples_split,
                        min_samples_leaf=self._min_samples_leaf)
            tree.fit(x, y)
            self._trees.append(tree)

    def eval(self, x, y):
        """"Evaluate accuracy on dataset."""
        p = self.predict(x)
        return np.sum(p == y) / x.shape[0]

    def predict(self, x):
        """Return predicted labels for given inputs."""
        return np.array([self._aggregate(x[i]) for i in range(x.shape[0])])

    def _aggregate(self, x):
        """Predict class by pooling predictions from all trees.
        Args:
            x(ndarray): A single example.
        Returns:
            (int): Predicted class index.
        """
        temp = [t.predict(x) for t in self._trees]
        _classes, counts = np.unique(np.array(temp), return_counts=True)

        # Return class with max count
        return _classes[np.argmax(counts)]

    def node_count(self):
        """Return number of nodes in forest."""
        return np.sum([t.node_count() for t in self._trees])

class KNN:
    def __init__(self, X, y, K=20):
        self.X = X
        self.y = y
        self.K = K

    def euclidian_distance(self, v1, v2):
        return np.linalg.norm(v1-v2)

    def predict(self, new_sample):
        # Initializing dict of distances and variable with size of training set
        distances, train_length = {}, len(self.X)

        # Calculating the Euclidean distance between the new
        # sample and the values of the training sample
        for i in range(train_length):
            d = self.euclidian_distance(self.X[i], new_sample)
            distances[i] = d
        
        # Selecting the K nearest neighbors
        k_neighbors = sorted(distances, key=distances.get)[:self.K]
        
        # Initializing labels counters
        ycounter = np.zeros(len(self.y[0]))
        for index in k_neighbors:
            ycounter += self.y[index]
        return np.exp(ycounter)/sum(np.exp(ycounter))
 

def serialize_logreg(model):
    # self.learning_rate = learning_rate  # learning_rate of the algorithm
    # self.num_iter = num_iter  #  number of iterations of the gradient descent
    # self.fit_intercept = fit_intercept  # boolean indicating whether we`re adding base X0 feature vector or not
    # self.verbose = verbose 
    # self._weights
    serialized_model = {
        'learning_rate':model.learning_rate,
        'num_iter':model.num_iter,
        'fit_intercept':model.fit_intercept,
        'verbose':model.verbose,
        'weights':model._weights.tolist()
    }
    return serialized_model

def deserialize_logreg(model_dict):
    deserialized_model = LogisticRegression()
    deserialized_model.learning_rate = model_dict['learning_rate']
    deserialized_model.num_iter = model_dict['num_iter']
    deserialized_model.fit_intercept = model_dict['fit_intercept']
    deserialized_model.verbose = model_dict['verbose']
    deserialized_model._weights = np.array(model_dict['weights'])
    return deserialized_model

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iter=100, fit_intercept=True, verbose=False):
        self.learning_rate = learning_rate  # learning_rate of the algorithm
        self.num_iter = num_iter  #  number of iterations of the gradient descent
        self.fit_intercept = fit_intercept  # boolean indicating whether we`re adding base X0 feature vector or not
        self.verbose = verbose  

    def _add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))  #  creating X0 features vector(M x 1)
        return np.concatenate((intercept, X), axis=1)  # concatenating X0 features vector with our features making intercept

    def _sigmoid(self, z):
        '''Defines our "logit" function based on which we make predictions
           parameters:
              z - product of the out features with weights
           return:
              probability of the attachment to class
        '''

        return 1 / (1 + np.exp(-z))

    def _loss(self, h, y):
        '''
        Functions have parameters or weights and we want to find the best values for them.
        To start we pick random values and we need a way to measure how well the algorithm performs using those random weights.
        That measure is computed using the loss function
        '''

        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    def get_params(self):
        return self._weights
    
    def load_params(self, params):
        self._weights = params

    def train(self, X, y):
        '''
        Function for training the algorithm.
            parameters:
              X - input data matrix (all our features without target variable)
              y - target variable vector (1/0)
            
            return:
              None
        '''

        if type(X) == pd.DataFrame:
            X = np.asarray(X)

        if self.fit_intercept:
            X = self._add_intercept(X)  # X will get a result with "zero" feature

        self._weights = np.zeros(X.shape[1])  #  inicializing our weights vector filled with zeros
        
        for i in range(self.num_iter):  # implementing Gradient Descent algorithm
            z = np.dot(X, self._weights)  #  calculate the product of the weights and predictor matrix
            h = self._sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self._weights -= self.learning_rate * gradient
            
            if (self.verbose == True and i % 10000 == 0):
                z = np.dot(X, self._weights)
                h = self._sigmoid(z)
                print(f'loss: {self._loss(h, y)} \t')

    def predict_prob(self, X):
        if type(X) == pd.DataFrame:
            X = np.asarray(X)
        if self.fit_intercept:
            X = self._add_intercept(X)
    
        return self._sigmoid(np.dot(X, self._weights))
    
    def predict(self, X, threshold=0.5):
        if type(X) == pd.DataFrame:
            X = np.asarray(X)

        return self.predict_prob(X) >= threshold
    
    def eval(self, X, y):
        """"Evaluate accuracy on dataset."""
        p = self.predict(X)
        return np.sum(p == y) / X.shape[0]

def wavg_bid_price(states, windows, product):
    ret = []
    mw = max(windows)
    ds = states[-mw:]
    wsum = 0
    vsum = 0
    avg = 0
    # print(f"wavg_bid_price ds: {ds}")
    for wi, d in enumerate(ds[::-1]):
        for p, v in d.order_depths[product].buy_orders.items():
            if not np.isnan(p):
                wsum += p*v
                vsum += v
        if wi+1 in windows:
            ret.append(wsum/vsum)
            avg += wsum/vsum

    while len(ret) < len(windows):
        ret.append(avg/len(windows))
    return ret

def wavg_ask_price(states, windows, product):
    ret = []
    mw = max(windows)
    ds = states[-mw:]
    wsum = 0
    vsum = 0
    avg = 0
    for wi, d in enumerate(ds[::-1]):
        for p, v in d.order_depths[product].sell_orders.items():
            if not np.isnan(p):
                wsum += p*v
                vsum += v
        if wi+1 in windows:
            ret.append(wsum/vsum)
            avg += wsum/vsum
    while len(ret) < len(windows):
        ret.append(avg/len(windows))
    return ret

def avg_mid_price(states, windows, product):
    ret = []
    mw = max(windows)
    ds = states[-mw:]
    wsum = 0
    vsum = 0
    avg = 0
    for wi, d in enumerate(ds[::-1]):
        order_book = d.order_depths[product]
        best_bid, best_ask = 0, 0
        if len(order_book.buy_orders) > 0:
            best_bid = max(order_book.buy_orders.keys())
        if len(order_book.sell_orders) > 0:
            best_ask = min(order_book.sell_orders.keys())
        
        if best_bid == 0:
            mid_price = best_ask
        elif best_ask == 0:
            mid_price = best_bid
        else:
            mid_price = best_bid + (best_ask-best_bid) / 2
        


        wsum += mid_price
        vsum += 1
        if wi+1 in windows:
            ret.append(wsum/vsum)
            avg += wsum/vsum
    while len(ret) < len(windows):
        ret.append(avg/len(windows))
    return ret

def volume_diff(states, windows, product):
    ret = []
    mw = max(windows)
    ds = states[-mw:]
    vsum = 0
    avg = 0
    for wi, d in enumerate(ds[::-1]):
        for p, v in d.order_depths[product].buy_orders.items():
            if not np.isnan(p):
                vsum -= v
        for p, v in d.order_depths[product].sell_orders.items():
            if not np.isnan(p):
                vsum += v
        if wi+1 in windows:
            ret.append(vsum)
            avg = vsum
    while len(ret) < len(windows):
        ret.append(0)
    return ret

def best_prices(states, windows, product):
    ret = []
    mw = max(windows)
    ds = states[-mw:]
    asks = []
    bids = []
    for wi, d in enumerate(ds[::-1]):
        for p, _ in d.order_depths[product].buy_orders.items():
            if not np.isnan(p):
                bids.append(p)
        for p, _ in d.order_depths[product].sell_orders.items():
            if not np.isnan(p):
                asks.append(p)
        if wi+1 in windows:
            ret.append(max(bids))
            ret.append(min(asks))
    while len(ret) < len(windows)*2:
        ret.append(0)

    return ret

def compute_single(states, inds, windows, product):
    x = []
    for ind in inds:
        x.extend(ind(states, windows, product))
    return x

class Trader:
    def __init__(self):
        self.states = []
        self.products = ["BANANAS", "PEARLS"]
        self.inds = [wavg_bid_price, wavg_ask_price, avg_mid_price, volume_diff, best_prices]
        self.windows = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        self.clfs = {product:{} for product in self.products}
        for product in self.products:
            logregs = [
                deserialize_logreg(model_dict) for model_dict in log_reg_hardcoded_params[product]
            ]
            self.clfs[product]['logregs'] = logregs
        for product in self.products:
            dectrees = [
                deserialize_decision_tree(model_dict) for model_dict in dec_tree_hardcoded_params[product]
            ]
            self.clfs[product]['dectrees'] = dectrees
        # for product in self.products:
        #     knn = KNN(xss[product], np.array(gts[product]))
        #     self.clfs['knn'] = knn

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        # Initialize the method output dict as an empty dict
        result = {}
        self.states.append(state)

        # Iterate over all the keys (the available products) contained in the order depths
        for product in self.products:
            x = np.array(compute_single(self.states, self.inds, self.windows, product))
            x = x[None, :]
            print(f"x for {product} made")

            logreg_out = [clf.predict(x) for clf in self.clfs[product]['logregs']]
            print(f"log_reg for {product} made")
            dectree_out = [clf.predict(x) for clf in self.clfs[product]['dectrees']]
            print(f"decision tree for {product} made")

            # knn_out = self.clfs[product]['knn'].predict(x)
            vote_gt = np.argmax([o1+o2 for o1,o2 in zip(logreg_out, dectree_out)])
            print(f"vote for {product}: {vote_gt}")
            # The GT is converted to classification with 7 classes - [MustBuy StrongBuy SoftBuy Neutral SoftSell StrongSell MustSell]
            # Look at trader_test for definitions
            # Hold is everything else

            
            # TODO: Make purchasing decision based on vote_gt

            orders: List[Order] = []
            order_depth: OrderDepth = state.order_depths[product]
            if vote_gt == 0: # Must Buy
                if len(order_depth.buy_orders) != 0:
                    best_ask = min(order_depth.sell_orders.keys())
                    best_ask_volume = order_depth.sell_orders[best_ask]
                    orders.append(Order(product, best_ask, 3))
                    # purchasing policy
            elif vote_gt == 1: # Strong Buy
                if len(order_depth.buy_orders) != 0:
                    best_ask = min(order_depth.sell_orders.keys())
                    best_ask_volume = order_depth.sell_orders[best_ask]
                    orders.append(Order(product, best_ask, 2))
                # purchasing policy
            elif vote_gt == 2: # Soft Buy
                if len(order_depth.buy_orders) != 0:
                    best_ask = min(order_depth.sell_orders.keys())
                    best_ask_volume = order_depth.sell_orders[best_ask]
                    orders.append(Order(product, best_ask, 1))
                # purchasing policy
            elif vote_gt == 3: # Must Sell/Short
                if len(order_depth.sell_orders) > 0:
                    best_bid = max(order_depth.buy_orders.keys())
                    best_ask_volume = order_depth.buy_orders[best_bid]
                    orders.append(Order(product, best_bid, -3))
                # purchasing policy
            elif vote_gt == 4: # Strong Sell/Short
                 if len(order_depth.sell_orders) > 0:
                    best_bid = max(order_depth.buy_orders.keys())
                    best_ask_volume = order_depth.buy_orders[best_bid]
                    orders.append(Order(product, best_bid, -2))
                # purchasing policy
            elif vote_gt == 5: # Soft Sell/Short
                 if len(order_depth.sell_orders) > 0:
                    best_bid = max(order_depth.buy_orders.keys())
                    best_ask_volume = order_depth.buy_orders[best_bid]
                    orders.append(Order(product, best_bid, -1))
                # purchasing policy
            elif vote_gt == 6: # Neutral
                pass # purchasing policy



            result[product] = orders
        
        


        return result