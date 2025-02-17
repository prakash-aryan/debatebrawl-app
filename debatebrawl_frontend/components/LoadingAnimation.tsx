// components/LoadingAnimation.tsx
import { Box, Spinner, Text, VStack } from '@chakra-ui/react';

const LoadingAnimation = () => {
  return (
    <Box
      position="fixed"
      top="0"
      left="0"
      right="0"
      bottom="0"
      backgroundColor="rgba(0, 0, 0, 0.7)"
      display="flex"
      alignItems="center"
      justifyContent="center"
      zIndex="9999"
    >
      <VStack spacing={4}>
        <Spinner size="xl" color="blue.500" thickness="4px" />
        <Text color="white" fontSize="xl" fontWeight="bold">
          Loading your debate...
        </Text>
      </VStack>
    </Box>
  );
};

export default LoadingAnimation;