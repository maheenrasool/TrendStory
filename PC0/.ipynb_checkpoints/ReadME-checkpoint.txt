"""
📡 Windows Server Setup: Allow MacBook Client to Access gRPC Server on Port 50052

This setup ensures your MacBook can communicate with your Windows (Acer) server
over Wi-Fi for gRPC and allows ping tests for debugging.

🔐 STEP 1: Allow Ping (ICMP) on Windows Firewall
-----------------------------------------------
1. Open 'Windows Defender Firewall' > Click 'Advanced Settings'
2. Go to 'Inbound Rules' > Click 'New Rule...'
3. Choose 'Custom' > Next
4. Program: Select 'All Programs' > Next
5. Protocol: Choose 'ICMPv4' > Next
6. Scope: Leave default > Next
7. Action: 'Allow the connection' > Next
8. Profile: Check ONLY 'Private'
9. Name: Allow Ping from Mac
10. Click 'Finish'

🔓 STEP 2: Allow gRPC Port 50052
-------------------------------
1. Go to 'Inbound Rules' > Click 'New Rule...'
2. Rule Type: Select 'Port' > Next
3. Protocol: TCP > Port: 50052 > Next
4. Action: 'Allow the connection' > Next
5. Profile: Check ONLY 'Private'
6. Name: Allow gRPC Port 50052
7. Click 'Finish'

🧪 STEP 3: Test From MacBook Terminal
-------------------------------------
$ ping 192.168.18.107         # Should return successful replies
$ nc -zv 192.168.18.107 50052 # Should say: "Connection succeeded!"

✅ Once confirmed, run server.py on Acer and client.py on MacBook.
"""
