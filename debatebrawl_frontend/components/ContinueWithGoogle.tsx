'use client'

import React, { useState } from 'react';
import { Box, Button, VStack, Text, useToast, Heading } from '@chakra-ui/react';
import { useRouter } from 'next/navigation';
import { FcGoogle } from 'react-icons/fc';
import { signInWithGoogle } from '@/utils/auth';
import { createUserDocument, getUserDocument } from '@/utils/firestore';
import { FirebaseError } from 'firebase/app';

export default function ContinueWithGoogle() {
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const router = useRouter();
  const toast = useToast();

  const handleGoogleAuth = async () => {
    setLoading(true);
    setError(null);
    try {
      const user = await signInWithGoogle();
      if (user) {
        // Check if the user document already exists
        const existingUser = await getUserDocument(user.uid);
        
        if (!existingUser) {
          // If user doesn't exist, create a new user document
          await createUserDocument({
            uid: user.uid,
            email: user.email!,
            name: user.displayName || '',
            username: user.email!.split('@')[0], // Using email prefix as temporary username
            remainingFreeDebates: 2,
            totalDebates: 0,
            wins: 0,
            losses: 0,
            draws: 0,
          });
          toast({
            title: 'Account created',
            description: 'Welcome to DebateBrawl!',
            status: 'success',
            duration: 3000,
            isClosable: true,
          });
        } else {
          toast({
            title: 'Signed in',
            description: 'Welcome back to DebateBrawl!',
            status: 'success',
            duration: 3000,
            isClosable: true,
          });
        }
        router.push('/dashboard');
      }
    } catch (err) {
      console.error("Google authentication error:", err);
      if (err instanceof FirebaseError) {
        setError(err.message);
      } else {
        setError('An error occurred during authentication. Please try again.');
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box maxWidth="400px" margin="auto" mt={8}>
      <Heading as="h1" size="xl" textAlign="center" mb={8}>
        Welcome to DebateBrawl
      </Heading>
      <Box bg="gray.800" p={8} borderRadius="md" boxShadow="lg">
        <VStack spacing={4}>
          <Button 
            leftIcon={<FcGoogle />} 
            onClick={handleGoogleAuth} 
            width="full" 
            variant="outline" 
            bg="white" 
            color="black" 
            _hover={{ bg: 'gray.100' }} 
            isLoading={loading}
          >
            Continue with Google
          </Button>
          {error && <Text color="red.500" mt={2}>{error}</Text>}
        </VStack>
      </Box>
    </Box>
  );
}