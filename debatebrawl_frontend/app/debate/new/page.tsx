'use client'

import React, { useState, useEffect } from 'react';
import { Box, Container, Heading, VStack, Button, Text, useToast, Radio, RadioGroup, Flex, Spinner } from '@chakra-ui/react';
import { useRouter } from 'next/navigation';
import { auth } from '@/utils/firebase';
import { onAuthStateChanged, User } from 'firebase/auth';
import { isEmailVerified } from '@/utils/auth';
import { getTopics, startDebate } from '@/utils/api';
import LoadingAnimation from '@/components/LoadingAnimation';

export default function NewDebate() {
  const [user, setUser] = useState<User | null>(null);
  const [topics, setTopics] = useState<string[]>([]);
  const [selectedTopic, setSelectedTopic] = useState('');
  const [position, setPosition] = useState('');
  const [loading, setLoading] = useState(true);
  const [isStartingDebate, setIsStartingDebate] = useState(false);
  const router = useRouter();
  const toast = useToast();

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (currentUser) => {
      if (currentUser && isEmailVerified(currentUser)) {
        setUser(currentUser);
        fetchTopics(currentUser.uid);
      } else {
        router.push('/');
      }
    });

    return () => unsubscribe();
  }, [router]);

  const fetchTopics = async (userId: string) => {
    try {
      const fetchedTopics = await getTopics(userId);
      setTopics(fetchedTopics);
    } catch (error) {
      console.error('Failed to fetch topics:', error);
      toast({
        title: 'Error',
        description: 'Failed to load debate topics. Please try again.',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setLoading(false);
    }
  };

  const handleStartDebate = async () => {
    if (!user || !selectedTopic || !position) return;

    setIsStartingDebate(true);
    try {
      const response = await startDebate(user.uid, selectedTopic, position);
      if (response && response.debate_id) {
        router.push(`/debate/${response.debate_id}`);
      } else {
        throw new Error('Invalid response from startDebate');
      }
    } catch (error) {
      console.error('Failed to start debate:', error);
      toast({
        title: 'Error',
        description: 'Failed to start the debate. Please try again.',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
      setIsStartingDebate(false);
    }
  };

  if (loading || isStartingDebate) {
    return <LoadingAnimation />;
  }

  return (
    <Container maxW="container.md" py={10}>
      <VStack spacing={8} align="stretch">
        <Heading as="h1" size="xl" textAlign="center" mb={6}>Start a New Debate</Heading>
        
        <Box>
          <Text fontSize="xl" fontWeight="bold" mb={4}>Select a Topic:</Text>
          <RadioGroup onChange={setSelectedTopic} value={selectedTopic}>
            <VStack align="stretch" spacing={4}>
              {topics.map((topic, index) => (
                <Radio key={index} value={topic} colorScheme="blue" size="lg">
                  <Text fontSize="lg" fontWeight="medium" ml={2}>
                    {index + 1}. {topic}
                  </Text>
                </Radio>
              ))}
            </VStack>
          </RadioGroup>
        </Box>

        <Box>
          <Text fontSize="xl" fontWeight="bold" mb={4}>Choose Your Position:</Text>
          <RadioGroup onChange={setPosition} value={position}>
            <VStack align="stretch" spacing={4}>
              <Radio value="for" colorScheme="green" size="lg">
                <Text fontSize="lg" fontWeight="medium" ml={2}>For</Text>
              </Radio>
              <Radio value="against" colorScheme="red" size="lg">
                <Text fontSize="lg" fontWeight="medium" ml={2}>Against</Text>
              </Radio>
            </VStack>
          </RadioGroup>
        </Box>

        <Button
          colorScheme="blue"
          size="lg"
          onClick={handleStartDebate}
          isDisabled={!selectedTopic || !position}
          mt={8}
        >
          Start Debate with AI
        </Button>
      </VStack>
    </Container>
  );
}