'use client'

import React, { useState, useEffect, useCallback } from 'react';
import { useParams, useRouter } from 'next/navigation';
import { Box, Container, Heading, Text, Textarea, Button, VStack, HStack, useToast, Flex, Badge, Spinner, Card, CardHeader, CardBody, Divider } from '@chakra-ui/react';
import { getDebateState, submitArgument } from '@/utils/api';
import { auth } from '@/utils/firebase';
import { onAuthStateChanged } from 'firebase/auth';

const TOTAL_ROUNDS = 5;

interface DebateState {
  status: string;
  current_round: number;
  current_turn: string;
  scores: { user: number; ai: number };
  arguments: {
    [key: string]: {
      user: string;
      ai: string;
    };
  };
  topic: string;
  llm_suggestions: string[];
  ga_strategy: string | null;
  as_prediction: string | null;
  user_position: string;
  ai_position: string;
  evaluation_feedback: string;
}

export default function DebatePage() {
  const params = useParams();
  const router = useRouter();
  const debateId = params.id as string;
  const [userId, setUserId] = useState<string | null>(null);
  const [debateState, setDebateState] = useState<DebateState | null>(null);
  const [argument, setArgument] = useState('');
  const [isLoading, setIsLoading] = useState(true);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isGameOver, setIsGameOver] = useState(false);
  const toast = useToast();

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (user) => {
      if (user) {
        setUserId(user.uid);
      } else {
        router.push('/signin');
      }
    });

    return () => unsubscribe();
  }, [router]);

  const fetchDebateState = useCallback(async () => {
    if (!debateId) return;
    try {
      const state = await getDebateState(debateId);
      setDebateState(state as unknown as DebateState);
      setIsLoading(false);
      if (state.current_round > TOTAL_ROUNDS) {
        setIsGameOver(true);
      }
    } catch (error) {
      console.error('Error fetching debate state:', error);
      toast({
        title: 'Error',
        description: 'Failed to fetch debate state. Please try again.',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    }
  }, [debateId, toast]);

  useEffect(() => {
    fetchDebateState();
    const interval = setInterval(fetchDebateState, 5000);
    return () => clearInterval(interval);
  }, [fetchDebateState]);

  const handleSubmitArgument = useCallback(async () => {
    if (!userId || !debateId || isSubmitting) return;
    setIsSubmitting(true);
    try {
      await submitArgument(debateId, userId, argument);
      setArgument('');
      await fetchDebateState();
      toast({
        title: 'Argument Submitted',
        description: 'Your argument has been successfully submitted.',
        status: 'success',
        duration: 3000,
        isClosable: true,
      });
    } catch (error) {
      console.error('Error submitting argument:', error);
      toast({
        title: 'Error',
        description: 'Failed to submit argument. Please try again.',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setIsSubmitting(false);
    }
  }, [userId, debateId, argument, toast, fetchDebateState]);

  if (isLoading) {
    return (
      <Box bg="gray.900" minHeight="100vh" color="white">
        <Container maxW="container.xl" centerContent>
          <VStack spacing={8} mt={16}>
            <Spinner size="xl" />
            <Text mt={4}>Loading debate...</Text>
          </VStack>
        </Container>
      </Box>
    );
  }

  if (!debateState || !userId) {
    return (
      <Box bg="gray.900" minHeight="100vh" color="white">
        <Container maxW="container.xl" centerContent>
          <Text mt={4}>Failed to load debate. Please try again.</Text>
        </Container>
      </Box>
    );
  }

  return (
    <Box bg="gray.900" minHeight="100vh" color="white">
      <Container maxW="container.xl" py={10}>
        <VStack spacing={8} align="stretch">
          <Card bg="gray.800" color="white">
            <CardHeader>
              <Heading size="xl" textAlign="center">{debateState.topic}</Heading>
            </CardHeader>
            <CardBody>
              <Flex justify="space-between" align="center">
                <Badge colorScheme="blue" fontSize="md" p={2}>
                  Round: {debateState.current_round} / {TOTAL_ROUNDS}
                </Badge>
                <Badge colorScheme={debateState.current_turn === 'user' ? 'green' : 'red'} fontSize="md" p={2}>
                  {debateState.current_turn === 'user' ? 'Your Turn' : "AI's Turn"}
                </Badge>
              </Flex>
            </CardBody>
          </Card>

          {isGameOver ? (
            <Card bg="gray.800" color="white">
              <CardHeader>
                <Heading size="lg" textAlign="center">Game Over</Heading>
              </CardHeader>
              <CardBody>
                <Text fontSize="xl" textAlign="center">
                  Final Scores: You - {debateState.scores.user.toFixed(2)}, AI - {debateState.scores.ai.toFixed(2)}
                </Text>
                <Text fontSize="xl" textAlign="center" mt={4}>
                  {debateState.scores.user > debateState.scores.ai ? "You win!" : 
                   debateState.scores.user < debateState.scores.ai ? "AI wins!" : "It's a tie!"}
                </Text>
              </CardBody>
            </Card>
          ) : (
            <Flex direction={{ base: 'column', lg: 'row' }} justify="space-between" width="100%" gap={8}>
              <Box width={{ base: '100%', lg: '60%' }}>
                <VStack spacing={4} align="stretch">
                  <Card bg="gray.800" color="white">
                    <CardHeader>
                      <Heading size="md">Your Argument ({debateState.user_position})</Heading>
                    </CardHeader>
                    <CardBody>
                      <Textarea
                        value={argument}
                        onChange={(e) => setArgument(e.target.value)}
                        placeholder="Enter your argument here..."
                        bg="gray.700"
                        color="white"
                        isDisabled={debateState.current_turn !== 'user' || isSubmitting}
                        mb={2}
                        minHeight="200px"
                      />
                      <Button
                        onClick={handleSubmitArgument}
                        colorScheme="blue"
                        isDisabled={debateState.current_turn !== 'user' || isSubmitting}
                        width="100%"
                        isLoading={isSubmitting}
                        loadingText="Submitting"
                      >
                        Submit Argument
                      </Button>
                    </CardBody>
                  </Card>
                  <Card bg="gray.800" color="white">
                    <CardHeader>
                      <Heading size="md">AI's Argument ({debateState.ai_position})</Heading>
                    </CardHeader>
                    <CardBody>
                      {Object.entries(debateState.arguments).map(([round, args]) => (
                        <Text key={round} mb={2}>
                          Round {round}: {args.ai || "No argument yet"}
                        </Text>
                      ))}
                    </CardBody>
                  </Card>
                  {debateState.evaluation_feedback && (
                    <Card bg="gray.800" color="white">
                      <CardHeader>
                        <Heading size="md">Evaluation Feedback</Heading>
                      </CardHeader>
                      <CardBody>
                        <Text>{debateState.evaluation_feedback}</Text>
                      </CardBody>
                    </Card>
                  )}
                </VStack>
              </Box>

              <Box width={{ base: '100%', lg: '40%' }}>
                <VStack spacing={4} align="stretch">
                  <Card bg="gray.800" color="white">
                    <CardHeader>
                      <Heading size="sm">AI Suggestions</Heading>
                    </CardHeader>
                    <CardBody>
                      <VStack align="start" spacing={3}>
                        {debateState.llm_suggestions.map((suggestion, index) => (
                          <Button
                            key={`suggestion-${index}`}
                            onClick={() => setArgument(prev => prev + ' ' + suggestion)}
                            size="sm"
                            variant="outline"
                            colorScheme="blue"
                            whiteSpace="normal"
                            textAlign="left"
                            height="auto"
                            py={2}
                            width="100%"
                            justifyContent="flex-start"
                          >
                            <Text fontSize="sm">{suggestion}</Text>
                          </Button>
                        ))}
                      </VStack>
                    </CardBody>
                  </Card>

                  {debateState.ga_strategy && (
                    <Card bg="gray.800" color="white">
                      <CardHeader>
                        <Heading size="sm">GA Strategy Suggestion</Heading>
                      </CardHeader>
                      <CardBody>
                        <Text fontSize="sm">{debateState.ga_strategy}</Text>
                      </CardBody>
                    </Card>
                  )}
                  {debateState.as_prediction && (
                    <Card bg="gray.800" color="white">
                      <CardHeader>
                        <Heading size="sm">AS Prediction</Heading>
                      </CardHeader>
                      <CardBody>
                        <Text fontSize="sm">{debateState.as_prediction}</Text>
                      </CardBody>
                    </Card>
                  )}
                </VStack>
              </Box>
            </Flex>
          )}

          <Card bg="gray.800" color="white">
            <CardBody>
              <Flex justify="space-between" width="100%">
                <Badge fontSize="lg" colorScheme="green" p={2}>Your Score: {debateState.scores.user.toFixed(2)}</Badge>
                <Badge fontSize="lg" colorScheme="red" p={2}>AI's Score: {debateState.scores.ai.toFixed(2)}</Badge>
              </Flex>
            </CardBody>
          </Card>

          <Card bg="gray.800" color="white">
            <CardHeader>
              <Heading size="md">What are GA and AS?</Heading>
            </CardHeader>
            <CardBody>
              <Text>
                <strong>GA (Genetic Algorithm):</strong> This is an AI technique that mimics natural selection to evolve debate strategies. It suggests the best combination of ethos (credibility), pathos (emotional appeal), and logos (logical reasoning) for your arguments.
              </Text>
              <Divider my={4} />
              <Text>
                <strong>AS (Adversarial Search):</strong> This AI method predicts your opponent's next move based on the current state of the debate. It helps you anticipate and prepare for the AI's likely arguments.
              </Text>
            </CardBody>
          </Card>
        </VStack>
      </Container>
    </Box>
  );
}