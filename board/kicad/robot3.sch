EESchema Schematic File Version 4
EELAYER 30 0
EELAYER END
$Descr A4 11693 8268
encoding utf-8
Sheet 1 1
Title ""
Date ""
Rev ""
Comp ""
Comment1 ""
Comment2 ""
Comment3 ""
Comment4 ""
$EndDescr
$Comp
L Driver_Motor:L293D U1
U 1 1 5E08A3E9
P 5350 2750
F 0 "U1" H 4950 3750 50  0000 C CNN
F 1 "L293D" H 5750 3750 50  0000 C CNN
F 2 "Package_DIP:DIP-16_W7.62mm" H 5600 2000 50  0001 L CNN
F 3 "http://www.ti.com/lit/ds/symlink/l293.pdf" H 5050 3450 50  0001 C CNN
	1    5350 2750
	1    0    0    -1  
$EndComp
$Comp
L Connector_Generic:Conn_01x02 J2
U 1 1 5E09145E
P 6900 2200
F 0 "J2" H 6980 2192 50  0000 L CNN
F 1 "M1" H 6980 2101 50  0000 L CNN
F 2 "Connector_JST:JST_XH_B2B-XH-A_1x02_P2.50mm_Vertical" H 6900 2200 50  0001 C CNN
F 3 "~" H 6900 2200 50  0001 C CNN
	1    6900 2200
	1    0    0    -1  
$EndComp
$Comp
L Connector_Generic:Conn_01x02 J3
U 1 1 5E091759
P 6900 2800
F 0 "J3" H 6980 2792 50  0000 L CNN
F 1 "M2" H 6980 2701 50  0000 L CNN
F 2 "Connector_JST:JST_XH_B2B-XH-A_1x02_P2.50mm_Vertical" H 6900 2800 50  0001 C CNN
F 3 "~" H 6900 2800 50  0001 C CNN
	1    6900 2800
	1    0    0    -1  
$EndComp
$Comp
L Connector_Generic:Conn_01x02 J1
U 1 1 5E09209D
P 1200 3000
F 0 "J1" H 1200 2700 50  0000 C CNN
F 1 "BAT" H 1200 2800 50  0000 C CNN
F 2 "Connector_JST:JST_XH_B2B-XH-A_1x02_P2.50mm_Vertical" H 1200 3000 50  0001 C CNN
F 3 "~" H 1200 3000 50  0001 C CNN
	1    1200 3000
	-1   0    0    1   
$EndComp
$Comp
L Device:CP C1
U 1 1 5E093ADC
P 1750 2950
F 0 "C1" H 1868 2996 50  0000 L CNN
F 1 "100μF" H 1868 2905 50  0000 L CNN
F 2 "Capacitor_THT:CP_Radial_D5.0mm_P2.50mm" H 1788 2800 50  0001 C CNN
F 3 "~" H 1750 2950 50  0001 C CNN
	1    1750 2950
	1    0    0    -1  
$EndComp
Wire Wire Line
	1400 2900 1400 2800
Wire Wire Line
	1400 2800 1750 2800
Wire Wire Line
	1400 3000 1400 3100
Wire Wire Line
	1400 3100 1750 3100
$Comp
L power:VDC #PWR01
U 1 1 5E094E9F
P 1750 2800
F 0 "#PWR01" H 1750 2700 50  0001 C CNN
F 1 "VDC" H 1750 3075 50  0000 C CNN
F 2 "" H 1750 2800 50  0001 C CNN
F 3 "" H 1750 2800 50  0001 C CNN
	1    1750 2800
	1    0    0    -1  
$EndComp
Connection ~ 1750 2800
$Comp
L power:GND #PWR02
U 1 1 5E095CC0
P 1750 3100
F 0 "#PWR02" H 1750 2850 50  0001 C CNN
F 1 "GND" H 1755 2927 50  0000 C CNN
F 2 "" H 1750 3100 50  0001 C CNN
F 3 "" H 1750 3100 50  0001 C CNN
	1    1750 3100
	1    0    0    -1  
$EndComp
Connection ~ 1750 3100
Wire Wire Line
	4850 2150 4500 2150
Wire Wire Line
	4850 2350 4500 2350
Wire Wire Line
	4850 2550 4500 2550
Wire Wire Line
	4850 2750 4500 2750
Wire Wire Line
	4850 2950 4500 2950
Wire Wire Line
	4850 3150 4500 3150
$Comp
L power:GND #PWR05
U 1 1 5E099FC3
P 5000 6200
F 0 "#PWR05" H 5000 5950 50  0001 C CNN
F 1 "GND" H 5005 6027 50  0000 C CNN
F 2 "" H 5000 6200 50  0001 C CNN
F 3 "" H 5000 6200 50  0001 C CNN
	1    5000 6200
	1    0    0    -1  
$EndComp
$Comp
L power:GND #PWR08
U 1 1 5E09A29C
P 5200 6200
F 0 "#PWR08" H 5200 5950 50  0001 C CNN
F 1 "GND" H 5205 6027 50  0000 C CNN
F 2 "" H 5200 6200 50  0001 C CNN
F 3 "" H 5200 6200 50  0001 C CNN
	1    5200 6200
	1    0    0    -1  
$EndComp
$Comp
L power:GND #PWR012
U 1 1 5E09A3F7
P 5400 6200
F 0 "#PWR012" H 5400 5950 50  0001 C CNN
F 1 "GND" H 5405 6027 50  0000 C CNN
F 2 "" H 5400 6200 50  0001 C CNN
F 3 "" H 5400 6200 50  0001 C CNN
	1    5400 6200
	1    0    0    -1  
$EndComp
$Comp
L power:GND #PWR015
U 1 1 5E09A483
P 5600 6200
F 0 "#PWR015" H 5600 5950 50  0001 C CNN
F 1 "GND" H 5605 6027 50  0000 C CNN
F 2 "" H 5600 6200 50  0001 C CNN
F 3 "" H 5600 6200 50  0001 C CNN
	1    5600 6200
	1    0    0    -1  
$EndComp
$Comp
L power:GND #PWR010
U 1 1 5E09AA41
P 5250 3550
F 0 "#PWR010" H 5250 3300 50  0001 C CNN
F 1 "GND" H 5255 3377 50  0000 C CNN
F 2 "" H 5250 3550 50  0001 C CNN
F 3 "" H 5250 3550 50  0001 C CNN
	1    5250 3550
	1    0    0    -1  
$EndComp
$Comp
L power:GND #PWR014
U 1 1 5E09AACD
P 5450 3550
F 0 "#PWR014" H 5450 3300 50  0001 C CNN
F 1 "GND" H 5455 3377 50  0000 C CNN
F 2 "" H 5450 3550 50  0001 C CNN
F 3 "" H 5450 3550 50  0001 C CNN
	1    5450 3550
	1    0    0    -1  
$EndComp
$Comp
L Driver_Motor:L293D U2
U 1 1 5E08B35B
P 5300 5400
F 0 "U2" H 4900 6400 50  0000 C CNN
F 1 "L293D" H 5700 6400 50  0000 C CNN
F 2 "Package_DIP:DIP-16_W7.62mm" H 5550 4650 50  0001 L CNN
F 3 "http://www.ti.com/lit/ds/symlink/l293.pdf" H 5000 6100 50  0001 C CNN
	1    5300 5400
	1    0    0    -1  
$EndComp
Wire Wire Line
	4800 5400 4450 5400
Wire Wire Line
	4800 5600 4450 5600
Wire Wire Line
	4800 5800 4450 5800
Text Label 4500 2150 0    50   ~ 0
D26
Text Label 4500 2350 0    50   ~ 0
D27
Text Label 4500 2750 0    50   ~ 0
D22
Text Label 4500 2950 0    50   ~ 0
D23
Text Label 4500 2550 0    50   ~ 0
D25
Text Label 4500 3150 0    50   ~ 0
D21
Text Label 4450 5400 0    50   ~ 0
D18
Text Label 4450 5600 0    50   ~ 0
D19
$Comp
L power:VDC #PWR011
U 1 1 5E0A65B1
P 5400 4400
F 0 "#PWR011" H 5400 4300 50  0001 C CNN
F 1 "VDC" H 5400 4675 50  0000 C CNN
F 2 "" H 5400 4400 50  0001 C CNN
F 3 "" H 5400 4400 50  0001 C CNN
	1    5400 4400
	1    0    0    -1  
$EndComp
$Comp
L power:VDC #PWR013
U 1 1 5E0A69D1
P 5450 1750
F 0 "#PWR013" H 5450 1650 50  0001 C CNN
F 1 "VDC" H 5450 2025 50  0000 C CNN
F 2 "" H 5450 1750 50  0001 C CNN
F 3 "" H 5450 1750 50  0001 C CNN
	1    5450 1750
	1    0    0    -1  
$EndComp
$Comp
L power:VCC #PWR07
U 1 1 5E0A72BF
P 5200 4400
F 0 "#PWR07" H 5200 4250 50  0001 C CNN
F 1 "VCC" H 5217 4573 50  0000 C CNN
F 2 "" H 5200 4400 50  0001 C CNN
F 3 "" H 5200 4400 50  0001 C CNN
	1    5200 4400
	1    0    0    -1  
$EndComp
$Comp
L power:VCC #PWR09
U 1 1 5E0A7789
P 5250 1750
F 0 "#PWR09" H 5250 1600 50  0001 C CNN
F 1 "VCC" H 5267 1923 50  0000 C CNN
F 2 "" H 5250 1750 50  0001 C CNN
F 3 "" H 5250 1750 50  0001 C CNN
	1    5250 1750
	1    0    0    -1  
$EndComp
Wire Wire Line
	6700 2150 6700 2200
Wire Wire Line
	6700 2350 6700 2300
Wire Wire Line
	6700 2750 6700 2800
Wire Wire Line
	6700 2950 6700 2900
Wire Wire Line
	6700 5400 6700 5450
Wire Wire Line
	6700 5600 6700 5550
Wire Wire Line
	6700 3250 6450 3250
Wire Wire Line
	6700 3350 6450 3350
Wire Wire Line
	6700 3450 6450 3450
Wire Wire Line
	6700 4200 6450 4200
Wire Wire Line
	6700 4300 6450 4300
$Comp
L power:VCC #PWR017
U 1 1 5E0D8A5C
P 6450 3450
F 0 "#PWR017" H 6450 3300 50  0001 C CNN
F 1 "VCC" V 6468 3577 50  0000 L CNN
F 2 "" H 6450 3450 50  0001 C CNN
F 3 "" H 6450 3450 50  0001 C CNN
	1    6450 3450
	0    -1   -1   0   
$EndComp
$Comp
L power:GND #PWR018
U 1 1 5E0D8E8B
P 6450 3250
F 0 "#PWR018" H 6450 3000 50  0001 C CNN
F 1 "GND" V 6455 3122 50  0000 R CNN
F 2 "" H 6450 3250 50  0001 C CNN
F 3 "" H 6450 3250 50  0001 C CNN
	1    6450 3250
	0    1    1    0   
$EndComp
Text Label 6450 3350 0    50   ~ 0
D16
$Comp
L power:GND #PWR019
U 1 1 5E0DCC7D
P 6450 4200
F 0 "#PWR019" H 6450 3950 50  0001 C CNN
F 1 "GND" V 6455 4072 50  0000 R CNN
F 2 "" H 6450 4200 50  0001 C CNN
F 3 "" H 6450 4200 50  0001 C CNN
	1    6450 4200
	0    1    1    0   
$EndComp
$Comp
L power:VCC #PWR020
U 1 1 5E0DD120
P 6450 4300
F 0 "#PWR020" H 6450 4150 50  0001 C CNN
F 1 "VCC" V 6468 4427 50  0000 L CNN
F 2 "" H 6450 4300 50  0001 C CNN
F 3 "" H 6450 4300 50  0001 C CNN
	1    6450 4300
	0    -1   -1   0   
$EndComp
Text Label 6500 4400 0    50   ~ 0
D5
$Comp
L Connector_Generic:Conn_01x03 J6
U 1 1 5E0C5DC6
P 6900 4300
F 0 "J6" H 6980 4342 50  0000 L CNN
F 1 "Servo" H 6980 4251 50  0000 L CNN
F 2 "Connector_PinHeader_2.54mm:PinHeader_1x03_P2.54mm_Vertical" H 6900 4300 50  0001 C CNN
F 3 "~" H 6900 4300 50  0001 C CNN
	1    6900 4300
	1    0    0    -1  
$EndComp
$Comp
L Connector_Generic:Conn_01x02 J4
U 1 1 5E091B5B
P 6900 5450
F 0 "J4" H 6980 5442 50  0000 L CNN
F 1 "M3" H 6980 5351 50  0000 L CNN
F 2 "Connector_JST:JST_XH_B2B-XH-A_1x02_P2.50mm_Vertical" H 6900 5450 50  0001 C CNN
F 3 "~" H 6900 5450 50  0001 C CNN
	1    6900 5450
	1    0    0    -1  
$EndComp
$Comp
L Connector_Generic:Conn_01x03 J5
U 1 1 5E0C579D
P 6900 3350
F 0 "J5" H 6980 3392 50  0000 L CNN
F 1 "LED" H 6980 3301 50  0000 L CNN
F 2 "Connector_PinHeader_2.54mm:PinHeader_1x03_P2.54mm_Vertical" H 6900 3350 50  0001 C CNN
F 3 "~" H 6900 3350 50  0001 C CNN
	1    6900 3350
	1    0    0    -1  
$EndComp
Text Label 4450 5800 0    50   ~ 0
D17
$Comp
L power:VCC #PWR0101
U 1 1 5E0E9736
P 850 7400
F 0 "#PWR0101" H 850 7250 50  0001 C CNN
F 1 "VCC" H 868 7573 50  0000 C CNN
F 2 "" H 850 7400 50  0001 C CNN
F 3 "" H 850 7400 50  0001 C CNN
	1    850  7400
	-1   0    0    1   
$EndComp
$Comp
L power:PWR_FLAG #FLG0101
U 1 1 5E0E9B60
P 850 7400
F 0 "#FLG0101" H 850 7475 50  0001 C CNN
F 1 "PWR_FLAG" H 850 7573 50  0000 C CNN
F 2 "" H 850 7400 50  0001 C CNN
F 3 "~" H 850 7400 50  0001 C CNN
	1    850  7400
	1    0    0    -1  
$EndComp
$Comp
L power:PWR_FLAG #FLG0102
U 1 1 5E0EA136
P 1350 7400
F 0 "#FLG0102" H 1350 7475 50  0001 C CNN
F 1 "PWR_FLAG" H 1350 7573 50  0000 C CNN
F 2 "" H 1350 7400 50  0001 C CNN
F 3 "~" H 1350 7400 50  0001 C CNN
	1    1350 7400
	1    0    0    -1  
$EndComp
$Comp
L power:GND #PWR0102
U 1 1 5E0EA4DA
P 1350 7400
F 0 "#PWR0102" H 1350 7150 50  0001 C CNN
F 1 "GND" H 1355 7227 50  0000 C CNN
F 2 "" H 1350 7400 50  0001 C CNN
F 3 "" H 1350 7400 50  0001 C CNN
	1    1350 7400
	1    0    0    -1  
$EndComp
$Comp
L power:PWR_FLAG #FLG0103
U 1 1 5E0EE97A
P 1850 7400
F 0 "#FLG0103" H 1850 7475 50  0001 C CNN
F 1 "PWR_FLAG" H 1850 7573 50  0000 C CNN
F 2 "" H 1850 7400 50  0001 C CNN
F 3 "~" H 1850 7400 50  0001 C CNN
	1    1850 7400
	1    0    0    -1  
$EndComp
$Comp
L power:VDC #PWR0103
U 1 1 5E0EEEEE
P 1850 7400
F 0 "#PWR0103" H 1850 7300 50  0001 C CNN
F 1 "VDC" H 1850 7675 50  0000 C CNN
F 2 "" H 1850 7400 50  0001 C CNN
F 3 "" H 1850 7400 50  0001 C CNN
	1    1850 7400
	-1   0    0    1   
$EndComp
Wire Wire Line
	5850 2150 6700 2150
Wire Wire Line
	5850 2350 6700 2350
Wire Wire Line
	5850 2750 6700 2750
Wire Wire Line
	5850 2950 6700 2950
Wire Wire Line
	5800 5400 6700 5400
Wire Wire Line
	5800 5600 6700 5600
Wire Wire Line
	3200 4100 3000 4100
Wire Wire Line
	3200 4200 3000 4200
$Comp
L power:VDC #PWR0104
U 1 1 5E13FA64
P 3000 4100
F 0 "#PWR0104" H 3000 4000 50  0001 C CNN
F 1 "VDC" H 3000 4375 50  0000 C CNN
F 2 "" H 3000 4100 50  0001 C CNN
F 3 "" H 3000 4100 50  0001 C CNN
	1    3000 4100
	1    0    0    -1  
$EndComp
$Comp
L power:GND #PWR0106
U 1 1 5E140668
P 1250 4200
F 0 "#PWR0106" H 1250 3950 50  0001 C CNN
F 1 "GND" V 1250 4050 50  0000 R CNN
F 2 "" H 1250 4200 50  0001 C CNN
F 3 "" H 1250 4200 50  0001 C CNN
	1    1250 4200
	0    1    1    0   
$EndComp
$Comp
L power:VCC #PWR0108
U 1 1 5E1A03D0
P 1300 4150
F 0 "#PWR0108" H 1300 4000 50  0001 C CNN
F 1 "VCC" H 1317 4323 50  0000 C CNN
F 2 "" H 1300 4150 50  0001 C CNN
F 3 "" H 1300 4150 50  0001 C CNN
	1    1300 4150
	1    0    0    -1  
$EndComp
$Comp
L power:GND #PWR0110
U 1 1 5E1A1974
P 3000 4150
F 0 "#PWR0110" H 3000 3900 50  0001 C CNN
F 1 "GND" V 3005 4022 50  0000 R CNN
F 2 "" H 3000 4150 50  0001 C CNN
F 3 "" H 3000 4150 50  0001 C CNN
	1    3000 4150
	0    1    1    0   
$EndComp
Text Label 1300 4350 0    50   ~ 0
D15
Text Label 1300 4450 0    50   ~ 0
D2
Text Label 1300 4550 0    50   ~ 0
D4
Text Label 1300 4650 0    50   ~ 0
D16
Text Label 1300 4750 0    50   ~ 0
D17
Text Label 1300 4850 0    50   ~ 0
D5
Text Label 1300 4950 0    50   ~ 0
D18
Text Label 1300 5050 0    50   ~ 0
D19
Text Label 1300 5150 0    50   ~ 0
D21
Text Label 1300 5250 0    50   ~ 0
D3
Text Label 1300 5350 0    50   ~ 0
D1
Text Label 1300 5450 0    50   ~ 0
D22
Text Label 1300 5550 0    50   ~ 0
D23
Wire Wire Line
	3200 4300 3000 4300
Wire Wire Line
	3200 4400 3000 4400
Wire Wire Line
	3200 4500 3000 4500
Wire Wire Line
	3200 4600 3000 4600
Wire Wire Line
	3200 4700 3000 4700
Wire Wire Line
	3200 4800 3000 4800
Wire Wire Line
	3200 4900 3000 4900
Wire Wire Line
	3200 5000 3000 5000
Wire Wire Line
	3200 5100 3000 5100
Wire Wire Line
	3200 5200 3000 5200
Wire Wire Line
	3200 5300 3000 5300
Wire Wire Line
	3200 5400 3000 5400
Wire Wire Line
	3200 5500 3000 5500
Text Label 3000 4300 0    50   ~ 0
D13
Text Label 3000 4400 0    50   ~ 0
D12
Text Label 3000 4500 0    50   ~ 0
D14
Text Label 3000 4600 0    50   ~ 0
D27
Text Label 3000 4700 0    50   ~ 0
D26
Text Label 3000 4800 0    50   ~ 0
D25
Text Label 3000 4900 0    50   ~ 0
D33
Text Label 3000 5000 0    50   ~ 0
D32
Text Label 3000 5100 0    50   ~ 0
D35
Text Label 3000 5200 0    50   ~ 0
D34
Text Label 3000 5300 0    50   ~ 0
D39
Text Label 3000 5400 0    50   ~ 0
D36
Text Label 3000 5500 0    50   ~ 0
EN
Text Notes 1050 5350 0    50   ~ 0
TX0\n
Text Notes 1050 5250 0    50   ~ 0
RX0
$Comp
L power:GND #PWR06
U 1 1 5E09A9AE
P 5050 3550
F 0 "#PWR06" H 5050 3300 50  0001 C CNN
F 1 "GND" H 5055 3377 50  0000 C CNN
F 2 "" H 5050 3550 50  0001 C CNN
F 3 "" H 5050 3550 50  0001 C CNN
	1    5050 3550
	1    0    0    -1  
$EndComp
$Comp
L power:GND #PWR016
U 1 1 5E09AB59
P 5650 3550
F 0 "#PWR016" H 5650 3300 50  0001 C CNN
F 1 "GND" H 5655 3377 50  0000 C CNN
F 2 "" H 5650 3550 50  0001 C CNN
F 3 "" H 5650 3550 50  0001 C CNN
	1    5650 3550
	1    0    0    -1  
$EndComp
Wire Wire Line
	5550 3550 5650 3550
Wire Wire Line
	5150 3550 5050 3550
Wire Wire Line
	5000 6200 5100 6200
Wire Wire Line
	5500 6200 5600 6200
Text Notes 1050 4450 0    50   ~ 0
LED
Text Notes 2750 4400 0    50   ~ 0
BOOT
Text Notes 2750 4500 0    50   ~ 0
OPWM
Text Notes 1050 4350 0    50   ~ 0
BOOT
Text Notes 2750 5200 0    50   ~ 0
IN
Text Notes 2750 5100 0    50   ~ 0
IN
Text Notes 2750 5400 0    50   ~ 0
Vp
Text Notes 2750 5300 0    50   ~ 0
Vn
Wire Wire Line
	1250 4200 1300 4200
Wire Wire Line
	1300 4200 1300 4250
Text Notes 1050 4650 0    50   ~ 0
RX2
Text Notes 1050 4750 0    50   ~ 0
TX2
Text Notes 1050 4850 0    50   ~ 0
BOOT
Text Notes 1050 4550 0    50   ~ 0
BOOT
Wire Wire Line
	1500 4950 1300 4950
Wire Wire Line
	1500 4850 1300 4850
Wire Wire Line
	1500 4750 1300 4750
Wire Wire Line
	1500 4650 1300 4650
Wire Wire Line
	1500 4550 1300 4550
Wire Wire Line
	1500 4450 1300 4450
Wire Wire Line
	1500 4350 1300 4350
Wire Wire Line
	1500 4250 1300 4250
Wire Wire Line
	1500 5550 1300 5550
Wire Wire Line
	1500 5450 1300 5450
Wire Wire Line
	1500 5350 1300 5350
Wire Wire Line
	1500 5250 1300 5250
Wire Wire Line
	1500 5150 1300 5150
Wire Wire Line
	1500 5050 1300 5050
Wire Wire Line
	1500 4150 1300 4150
Wire Wire Line
	3000 4200 3000 4150
$Comp
L Connector_Generic:Conn_02x15_Odd_Even J8
U 1 1 5E2FF7EA
P 3400 4800
F 0 "J8" H 3450 5717 50  0000 C CNN
F 1 "ESP32_2" H 3450 5626 50  0000 C CNN
F 2 "Connector_PinSocket_2.54mm:PinSocket_2x15_P2.54mm_Vertical" H 3400 4800 50  0001 C CNN
F 3 "~" H 3400 4800 50  0001 C CNN
	1    3400 4800
	1    0    0    -1  
$EndComp
Text Label 3700 4300 0    50   ~ 0
D13
Text Label 3700 4400 0    50   ~ 0
D12
Text Label 3700 4500 0    50   ~ 0
D14
Text Label 3700 4600 0    50   ~ 0
D27
Text Label 3700 4700 0    50   ~ 0
D26
Text Label 3700 4800 0    50   ~ 0
D25
Text Label 3700 4900 0    50   ~ 0
D33
Text Label 3700 5000 0    50   ~ 0
D32
Text Label 3700 5100 0    50   ~ 0
D35
Text Label 3700 5200 0    50   ~ 0
D34
Text Label 3700 5300 0    50   ~ 0
D39
Text Label 3700 5400 0    50   ~ 0
D36
Text Label 3700 5500 0    50   ~ 0
EN
$Comp
L power:VCC #PWR0105
U 1 1 5E30FC5C
P 2000 4150
F 0 "#PWR0105" H 2000 4000 50  0001 C CNN
F 1 "VCC" H 2017 4323 50  0000 C CNN
F 2 "" H 2000 4150 50  0001 C CNN
F 3 "" H 2000 4150 50  0001 C CNN
	1    2000 4150
	1    0    0    -1  
$EndComp
$Comp
L power:GND #PWR0107
U 1 1 5E310129
P 3700 4150
F 0 "#PWR0107" H 3700 3900 50  0001 C CNN
F 1 "GND" V 3705 4022 50  0000 R CNN
F 2 "" H 3700 4150 50  0001 C CNN
F 3 "" H 3700 4150 50  0001 C CNN
	1    3700 4150
	0    -1   -1   0   
$EndComp
Wire Wire Line
	3700 4150 3700 4200
$Comp
L Connector_Generic:Conn_02x15_Odd_Even J7
U 1 1 5E316ACA
P 1700 4850
F 0 "J7" H 1750 5767 50  0000 C CNN
F 1 "ESP32_1" H 1750 5676 50  0000 C CNN
F 2 "Connector_PinSocket_2.54mm:PinSocket_2x15_P2.54mm_Vertical" H 1700 4850 50  0001 C CNN
F 3 "~" H 1700 4850 50  0001 C CNN
	1    1700 4850
	1    0    0    -1  
$EndComp
Text Label 2000 4350 0    50   ~ 0
D15
Text Label 2000 4450 0    50   ~ 0
D2
Text Label 2000 4550 0    50   ~ 0
D4
Text Label 2000 4650 0    50   ~ 0
D16
Text Label 2000 4750 0    50   ~ 0
D17
Text Label 2000 4850 0    50   ~ 0
D5
Text Label 2000 4950 0    50   ~ 0
D18
Text Label 2000 5050 0    50   ~ 0
D19
Text Label 2000 5150 0    50   ~ 0
D21
Text Label 2000 5250 0    50   ~ 0
D3
Text Label 2000 5350 0    50   ~ 0
D1
Text Label 2000 5450 0    50   ~ 0
D22
Text Label 2000 5550 0    50   ~ 0
D23
$Comp
L power:VDC #PWR0109
U 1 1 5E31D717
P 3700 4100
F 0 "#PWR0109" H 3700 4000 50  0001 C CNN
F 1 "VDC" H 3700 4375 50  0000 C CNN
F 2 "" H 3700 4100 50  0001 C CNN
F 3 "" H 3700 4100 50  0001 C CNN
	1    3700 4100
	1    0    0    -1  
$EndComp
$Comp
L power:GND #PWR0111
U 1 1 5E31DC0E
P 2000 4200
F 0 "#PWR0111" H 2000 3950 50  0001 C CNN
F 1 "GND" V 2005 4072 50  0000 R CNN
F 2 "" H 2000 4200 50  0001 C CNN
F 3 "" H 2000 4200 50  0001 C CNN
	1    2000 4200
	0    -1   -1   0   
$EndComp
Wire Wire Line
	2000 4200 2000 4250
NoConn ~ 4800 5200
NoConn ~ 4800 5000
NoConn ~ 4800 4800
NoConn ~ 5800 4800
NoConn ~ 5800 5000
Wire Wire Line
	6700 4400 6500 4400
Text Label 6100 2150 0    50   ~ 0
M1A
Text Label 6100 2350 0    50   ~ 0
M1B
Text Label 6100 2750 0    50   ~ 0
M2A
Text Label 6100 2950 0    50   ~ 0
M2B
Text Label 6100 5400 0    50   ~ 0
M3A
Text Label 6100 5600 0    50   ~ 0
M3B
$Comp
L Connector_Generic:Conn_01x04 J9
U 1 1 5E0E72E0
P 3200 2350
F 0 "J9" H 3280 2342 50  0000 L CNN
F 1 "I2C" H 3280 2251 50  0000 L CNN
F 2 "Connector_PinSocket_2.54mm:PinSocket_1x04_P2.54mm_Vertical" H 3200 2350 50  0001 C CNN
F 3 "~" H 3200 2350 50  0001 C CNN
	1    3200 2350
	1    0    0    -1  
$EndComp
$Comp
L power:GND #PWR0112
U 1 1 5E0E7DA4
P 3000 2250
F 0 "#PWR0112" H 3000 2000 50  0001 C CNN
F 1 "GND" V 3005 2122 50  0000 R CNN
F 2 "" H 3000 2250 50  0001 C CNN
F 3 "" H 3000 2250 50  0001 C CNN
	1    3000 2250
	0    1    1    0   
$EndComp
$Comp
L power:VCC #PWR0113
U 1 1 5E0E8303
P 3000 2350
F 0 "#PWR0113" H 3000 2200 50  0001 C CNN
F 1 "VCC" V 3018 2477 50  0000 L CNN
F 2 "" H 3000 2350 50  0001 C CNN
F 3 "" H 3000 2350 50  0001 C CNN
	1    3000 2350
	0    -1   -1   0   
$EndComp
Wire Wire Line
	3000 2450 2800 2450
Wire Wire Line
	3000 2550 2800 2550
Text Label 2800 2450 0    50   ~ 0
D12
Text Label 2800 2550 0    50   ~ 0
D14
$EndSCHEMATC