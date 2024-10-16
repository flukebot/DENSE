import React from 'react';
import { Box, Button, Heading, Text, VStack, HStack, Icon, Input } from '@chakra-ui/react';
import { FaNetworkWired, FaPlay, FaPause, FaStop, FaExclamationTriangle } from 'react-icons/fa';

// Declare the WebSocket URL using currentHost
const currentHost = window.location.hostname;
const wsUrl = `ws://${currentHost}:4125/ws`;

class OHome extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      status: 'Idle',
      shardInfo: 'No shard assigned',
      isRunning: false,
      socket: null,
      password: '',
      isPasswordSet: false,
      lstPods: new Map(),
      lstAiPodStatusUpdate: new Map(),
      sharedKey: null,
      clientKeyPair: null,
      serverPublicKey: null,
    };
  }

  handlePasswordChange = (event) => {
    this.setState({ password: event.target.value });
  };

  handleSetPassword = async () => {
    const { password } = this.state;
    if (password) {
      try {
        // Generate client's ECDH key pair
        const clientKeyPair = await window.crypto.subtle.generateKey(
          {
            name: "ECDH",
            namedCurve: "P-256",
          },
          true,
          ["deriveBits"]
        );

        this.setState({ isPasswordSet: true, clientKeyPair }, () => {
          this.initializeWebSocket();
        });
      } catch (error) {
        console.error("Key pair generation failed:", error);
        this.setState({ status: 'Key generation failed' });
      }
    }
  };

  initializeWebSocket = () => {
    try {
      const socket = new window.WebSocket(wsUrl);
      socket.binaryType = "arraybuffer";

      socket.onopen = () => {
        console.log('WebSocket connected');
        this.setState({ status: 'Connected to server', socket });
        // Server sends its public key first
      };

      socket.onmessage = async (evt) => {
        const message = evt.data;

        if (typeof message === 'string') {
          // Assume it's the server's public key in PEM format
          const serverPublicKey = await this.importServerPublicKey(message);
          this.setState({ serverPublicKey }, async () => {
            // Send client's public key to server
            await this.sendClientPublicKey();
            // Derive shared secret after key exchange
            await this.deriveSharedSecret();
          });
        } else {
          // Assume it's an encrypted message
          const decryptedData = await this.decryptMessage(message);
          if (decryptedData) {
            const parsedMessage = JSON.parse(decryptedData);
            this.handleWebSocketMessage(parsedMessage);
          }
        }
      };

      socket.onclose = () => {
        console.log('WebSocket disconnected');
        this.setState({ status: 'Disconnected' });
        // Optionally implement reconnection logic here
      };

      socket.onerror = (error) => {
        console.error('WebSocket error:', error);
        this.setState({ status: 'Connection error' });
      };

      this.setState({ socket });
    } catch (error) {
      console.error('Failed to create WebSocket:', error);
      this.setState({ status: 'WebSocket initialization failed' });
    }
  };

  importServerPublicKey = async (pem) => {
    try {
      // Remove PEM header and footer and all whitespace
      const pemHeader = "-----BEGIN PUBLIC KEY-----";
      const pemFooter = "-----END PUBLIC KEY-----";
      const pemContents = pem
        .replace(pemHeader, '')
        .replace(pemFooter, '')
        .replace(/\s/g, '');

      // Decode Base64 to binary string
      const binaryDerString = window.atob(pemContents);
      const binaryDer = this.str2ab(binaryDerString);

      // Import the server's public key
      const serverPublicKey = await window.crypto.subtle.importKey(
        "spki",
        binaryDer,
        {
          name: "ECDH",
          namedCurve: "P-256",
        },
        true,
        []
      );

      return serverPublicKey;
    } catch (error) {
      console.error("Failed to import server public key:", error);
      this.setState({ status: 'Failed to import server public key' });
      return null;
    }
  };

  sendClientPublicKey = async () => {
    const { socket, clientKeyPair } = this.state;
    if (!socket || socket.readyState !== window.WebSocket.OPEN) {
      this.setState({ status: 'Unable to send public key: Not connected' });
      return;
    }

    try {
      // Export client's public key
      const exportedPubKey = await window.crypto.subtle.exportKey(
        "spki",
        clientKeyPair.publicKey
      );

      // Convert ArrayBuffer to base64 string
      const binary = String.fromCharCode(...new Uint8Array(exportedPubKey));
      const base64PubKey = window.btoa(binary);

      // Create PEM format
      const pemHeader = "-----BEGIN PUBLIC KEY-----\n";
      const pemFooter = "\n-----END PUBLIC KEY-----\n";
      const pemBody = base64PubKey.match(/.{1,64}/g).join('\n');
      const pem = `${pemHeader}${pemBody}${pemFooter}`;

      // Send client's public key to server as TextMessage
      socket.send(pem);
    } catch (error) {
      console.error("Failed to send client public key:", error);
      this.setState({ status: 'Failed to send public key' });
    }
  };

  deriveSharedSecret = async () => {
    const { clientKeyPair, serverPublicKey } = this.state;
    if (!clientKeyPair || !serverPublicKey) {
      console.error('Key pairs not set');
      this.setState({ status: 'Key pairs not set' });
      return;
    }

    try {
      // Derive shared secret as bits
      const sharedSecretBits = await window.crypto.subtle.deriveBits(
        {
          name: "ECDH",
          public: serverPublicKey,
        },
        clientKeyPair.privateKey,
        256 // bits to derive
      );

      // Hash the shared secret with SHA-256
      const sharedSecretHash = await window.crypto.subtle.digest(
        "SHA-256",
        sharedSecretBits
      );

      // Import the hash as AES-GCM key
      const sharedKey = await window.crypto.subtle.importKey(
        "raw",
        sharedSecretHash,
        {
          name: "AES-GCM",
        },
        false,
        ["encrypt", "decrypt"]
      );

      this.setState({ sharedKey }, () => {
        // Authenticate after deriving shared key
        this.authenticate();
      });
    } catch (error) {
      console.error("Failed to derive shared secret:", error);
      this.setState({ status: 'Failed to derive shared secret' });
    }
  };

  authenticate = async () => {
    const { socket, sharedKey, password } = this.state;
    if (!socket || socket.readyState !== window.WebSocket.OPEN || !sharedKey) {
      this.setState({ status: 'Unable to authenticate: Not connected or key not established' });
      return;
    }

    try {
      // Create authentication message
      const authMessage = { type: 'ping', password: password };
      const messageJSON = JSON.stringify(authMessage);

      // Encrypt the message
      const encoder = new TextEncoder();
      const data = encoder.encode(messageJSON);

      const iv = window.crypto.getRandomValues(new Uint8Array(12)); // 96-bit nonce for AES-GCM

      const ciphertext = await window.crypto.subtle.encrypt(
        {
          name: "AES-GCM",
          iv: iv,
        },
        sharedKey,
        data
      );

      // Concatenate IV and ciphertext
      const encryptedMessage = new Uint8Array(iv.length + ciphertext.byteLength);
      encryptedMessage.set(iv, 0);
      encryptedMessage.set(new Uint8Array(ciphertext), iv.length);

      // Send encrypted message as binary
      socket.send(encryptedMessage.buffer);

      // Set status to awaiting authentication response
      this.setState({ status: 'Authenticating...' });
    } catch (error) {
      console.error("Authentication failed:", error);
      this.setState({ status: 'Authentication failed' });
    }
  };

  encryptMessage = async (plaintext) => {
    const { sharedKey } = this.state;
    if (!sharedKey) {
      console.error('Shared key is not established');
      this.setState({ status: 'Shared key not established' });
      return null;
    }

    try {
      const encoder = new TextEncoder();
      const data = encoder.encode(plaintext);

      const iv = window.crypto.getRandomValues(new Uint8Array(12)); // 96-bit nonce for AES-GCM

      const ciphertext = await window.crypto.subtle.encrypt(
        {
          name: "AES-GCM",
          iv: iv,
        },
        sharedKey,
        data
      );

      // Concatenate IV and ciphertext
      const encryptedMessage = new Uint8Array(iv.length + ciphertext.byteLength);
      encryptedMessage.set(iv, 0);
      encryptedMessage.set(new Uint8Array(ciphertext), iv.length);

      return encryptedMessage.buffer;
    } catch (error) {
      console.error('Encryption failed:', error);
      return null;
    }
  };

  decryptMessage = async (encryptedMessage) => {
    const { sharedKey } = this.state;
    if (!sharedKey) {
      console.error('Shared key is not established');
      this.setState({ status: 'Shared key not established' });
      return null;
    }

    try {
      const encryptedArray = new Uint8Array(encryptedMessage);
      const iv = encryptedArray.slice(0, 12);
      const ciphertext = encryptedArray.slice(12);

      const decrypted = await window.crypto.subtle.decrypt(
        {
          name: "AES-GCM",
          iv: iv,
        },
        sharedKey,
        ciphertext
      );

      const decoder = new TextDecoder();
      return decoder.decode(decrypted);
    } catch (error) {
      console.error('Decryption failed:', error);
      this.setState({ status: 'Decryption failed' });
      return null;
    }
  };

  str2ab = (str) => {
    const buf = new ArrayBuffer(str.length);
    const bufView = new Uint8Array(buf);
    for (let i = 0; i < str.length; i++) {
      bufView[i] = str.charCodeAt(i);
    }
    return buf;
  };

  handleWebSocketMessage = (message) => {
    switch (message.msgType) {
      case 'pong':
        this.setState({ status: 'Authenticated', isRunning: false });
        break;
      case 'auth_error':
        this.setState({ status: 'Authentication failed' });
        this.state.socket.close();
        break;
      // Handle other message types as needed
      default:
        console.log('Unknown message type:', message);
    }
  };

  sendEncryptedMessage = async (msgObj) => {
    const { socket, sharedKey } = this.state;
    if (!socket || socket.readyState !== window.WebSocket.OPEN || !sharedKey) {
      this.setState({ status: 'Unable to send message: Not connected or key not established' });
      return;
    }

    try {
      const messageJSON = JSON.stringify(msgObj);
      const encryptedMessage = await this.encryptMessage(messageJSON);
      if (encryptedMessage) {
        socket.send(encryptedMessage);
      }
    } catch (error) {
      console.error('Failed to send encrypted message:', error);
      this.setState({ status: 'Failed to send message' });
    }
  };

  startTraining = () => {
    this.sendEncryptedMessage({ type: 'start' });
    this.setState({ isRunning: true, status: 'Training started' });
  };

  pauseTraining = () => {
    this.sendEncryptedMessage({ type: 'pause' });
    this.setState({ isRunning: false, status: 'Training paused' });
  };

  stopTraining = () => {
    this.sendEncryptedMessage({ type: 'stop' });
    this.setState({ isRunning: false, status: 'Training stopped' });
  };

  render() {
    return (
      <VStack spacing={8} p={10} align="center">
        <Heading as="h1" size="xl">DENSE: Distributed Evolutionary Network Search Engine</Heading>
        <Text fontSize="lg" textAlign="center">
          Contribute your computational power to our evolving neural network. Note: By participating, you agree that this software is provided "as is" and we are not responsible for any potential damage to your machine. This will utilize 100% of your CPU power.
        </Text>
        {!this.state.isPasswordSet ? (
          <VStack spacing={4}>
            <Input
              placeholder="Enter password to proceed"
              type="password"
              value={this.state.password}
              onChange={this.handlePasswordChange}
            />
            <Button colorScheme="blue" onClick={this.handleSetPassword}>
              Set Password
            </Button>
          </VStack>
        ) : (
          <Box p={5} shadow="md" borderWidth="1px" width="100%" maxW="lg">
            <HStack spacing={4} mb={4}>
              <Icon as={FaNetworkWired} boxSize={6} />
              <Text fontSize="lg" fontWeight="bold">Status: {this.state.status}</Text>
            </HStack>
            <HStack spacing={4} mb={4}>
              <Text fontSize="md">Shard Info: {this.state.shardInfo}</Text>
            </HStack>
            <HStack spacing={6}>
              <Button
                colorScheme="green"
                leftIcon={<FaPlay />}
                onClick={this.startTraining}
                isDisabled={this.state.isRunning}
              >
                Start Training
              </Button>
              <Button
                colorScheme="yellow"
                leftIcon={<FaPause />}
                onClick={this.pauseTraining}
                isDisabled={!this.state.isRunning}
              >
                Pause Training
              </Button>
              <Button
                colorScheme="red"
                leftIcon={<FaStop />}
                onClick={this.stopTraining}
              >
                Stop Training
              </Button>
            </HStack>
          </Box>
        )}
        <Box bg="yellow.100" p={5} borderRadius="md" borderWidth="1px">
          <HStack spacing={4}>
            <Icon as={FaExclamationTriangle} color="yellow.600" boxSize={6} />
            <Text fontSize="md">
              Disclaimer: You are contributing to the DENSE neural network training, which may utilize 100% of your CPU. This software is provided as-is, and we are not responsible for any potential issues or damage.
            </Text>
          </HStack>
        </Box>
      </VStack>
    );
  }
}

export default OHome;
